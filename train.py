"""
Fighter Training Script (BC + PPO) - Compatible with Enhanced Simulation

Usage:
    python train_fighter.py

Pipeline:
    1. Behavioral Cloning (BC) pretraining from heuristic expert
    2. PPO reinforcement learning to improve evasion
    3. Saves fighter_policy.pt for use in main simulation
"""

import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Import enhanced simulation
import missle_sim as sim

# Use config from main sim
CFG = sim.CFG
ATMOS = sim.ATMOS

TRAINING_MISSILE_INITIAL_SPEED = 300.0  # m/s


# ===========================================================================
# Training Environment
# ===========================================================================

class FighterEnv:
    """
    RL training environment: fighter vs PN-guided missile.
    
    Observation: 14D vector (relative geometry, velocities, altitude)
    Action: 4D (lateral direction, vertical direction, unused, magnitude)
    """

    def __init__(self, dt=0.02, max_time=40.0, seed=0):
        self.dt = dt
        self.max_steps = int(max_time / dt)
        self.rng = np.random.default_rng(seed)
        self.alt_band = (9000.0, 13000.0)
        self.kill_radius = CFG.KILL_DIST
        self.step_count = 0
        self.altitude_cmd = None
        self.last_alt_update = 0.0
        self.prev_fighter_vel = None

    def reset(self):
        """Reset environment with random geometry."""
        ac_start, mi_start = sim.sample_random_starts(self.rng)
        
        self.fighter_pos = ac_start.astype(float).copy()
        self.fighter_vel = np.array([CFG.TARG_INITIAL_SPEED, 0.0, 0.0], dtype=float)
        self.missile_pos = mi_start.astype(float).copy()
        
        # Point missile at fighter
        rel = self.fighter_pos - self.missile_pos
        R0 = np.linalg.norm(rel)
        los = rel / max(R0, 1e-6)
        self.missile_vel = TRAINING_MISSILE_INITIAL_SPEED * los
        
        self.step_count = 0
        self.altitude_cmd = np.clip(self.fighter_pos[2], *self.alt_band)
        self.last_alt_update = 0.0
        self.prev_fighter_vel = None
        
        return self._get_obs()

    def _get_obs(self):
        """Build 14D observation vector."""
        r = self.missile_pos - self.fighter_pos
        v_rel = self.missile_vel - self.fighter_vel
        R = np.linalg.norm(r)
        speed = np.linalg.norm(self.fighter_vel)
        
        alt_mid = sum(self.alt_band) / 2
        alt_span = self.alt_band[1] - self.alt_band[0]
        
        obs = np.concatenate([
            r / 30000.0,                    # 3: relative position
            v_rel / 1000.0,                 # 3: relative velocity
            self.fighter_vel / 400.0,       # 3: own velocity
            np.array([
                R / 30000.0,                # range
                speed / 400.0,              # speed
                (self.fighter_pos[2] - alt_mid) / (alt_span / 2),  # altitude
                self.fighter_vel[2] / 200.0,  # vertical velocity
                self.fighter_vel[2] / max(speed, 1e-3)  # pitch
            ], dtype=float)
        ])
        return obs.astype(np.float32)

    def _action_to_accel(self, action):
        """Convert 4D action to world-frame acceleration."""
        a = np.asarray(action, dtype=float)
        dir_raw, mag_raw = a[:3], a[3]
        
        v = self.fighter_vel
        speed = np.linalg.norm(v)
        fwd = v / max(speed, 1e-6)
        
        # Build local coordinate frame
        world_up = np.array([0., 0., 1.])
        right = np.cross(fwd, world_up)
        r_norm = np.linalg.norm(right)
        right = right / max(r_norm, 1e-6) if r_norm > 1e-6 else np.array([0., 1., 0.])
        up_local = np.cross(right, fwd)
        
        # Limited vertical authority (realistic)
        d_r, d_u, _ = dir_raw
        d_u_limited = d_u * 0.4
        
        dir_local = d_r * right + d_u_limited * up_local
        d_norm = np.linalg.norm(dir_local)
        if d_norm < 1e-6:
            return np.zeros(3)
        
        dir_unit = dir_local / d_norm
        mag_scale = max(0.0, math.tanh(mag_raw))
        max_accel = CFG.TARG_MAX_G * 9.81
        
        return mag_scale * max_accel * dir_unit

    def _expert_accel_world(self, t):
        """Heuristic expert evasion strategy."""
        max_accel = CFG.TARG_MAX_G * 9.81
        
        r = self.missile_pos - self.fighter_pos
        R = np.linalg.norm(r)
        if R < 1e-6:
            return np.zeros(3)
        
        los = r / R
        speed = np.linalg.norm(self.fighter_vel)
        fwd = self.fighter_vel / max(speed, 1e-3)
        
        # Altitude control
        if t - self.last_alt_update > 5.0:
            self.altitude_cmd = self.rng.uniform(*self.alt_band)
            self.last_alt_update = t
        
        alt_error = self.altitude_cmd - self.fighter_pos[2]
        a_z = np.clip(0.01 * alt_error, -2*9.81, 2*9.81)
        vert_accel = np.array([0., 0., a_z])
        
        # Threat level
        threat = 1.0 if R < 8000 else 0.5 if R < 25000 else 0.2
        
        # Break perpendicular to LOS
        los_away = -los
        lateral = los_away - np.dot(los_away, fwd) * fwd
        lat_norm = np.linalg.norm(lateral)
        if lat_norm < 1e-5:
            lateral = np.cross(fwd, np.array([0., 0., 1.]))
            lat_norm = np.linalg.norm(lateral)
        lateral = lateral / max(lat_norm, 1e-6)
        
        lat_accel = threat * max_accel * 0.9 * lateral
        
        # Random jink
        jink = 0.2 * 9.81 * self.rng.standard_normal(3)
        jink -= np.dot(jink, fwd) * fwd
        
        a_cmd = vert_accel + lat_accel + jink
        
        # G-limit
        total = a_cmd + CFG.GRAVITY
        total_mag = np.linalg.norm(total)
        if total_mag > max_accel:
            total *= max_accel / total_mag
            a_cmd = total - CFG.GRAVITY
        
        return a_cmd

    def expert_action(self, t):
        """Convert expert accel to action space."""
        a_des = self._expert_accel_world(t)
        
        speed = np.linalg.norm(self.fighter_vel)
        fwd = self.fighter_vel / max(speed, 1e-6)
        
        world_up = np.array([0., 0., 1.])
        right = np.cross(fwd, world_up)
        r_norm = np.linalg.norm(right)
        right = right / max(r_norm, 1e-6) if r_norm > 1e-6 else np.array([0., 1., 0.])
        up_local = np.cross(right, fwd)
        
        a_r = np.dot(a_des, right)
        a_u = np.dot(a_des, up_local)
        mag = math.sqrt(a_r**2 + a_u**2)
        
        max_accel = CFG.TARG_MAX_G * 9.81
        mag_scale = np.clip(mag / max(max_accel, 1e-6), 0.0, 0.999)
        mag_raw = 0.5 * math.log((1 + mag_scale) / (1 - mag_scale + 1e-8))
        
        return np.array([a_r, a_u, 0.0, mag_raw], dtype=np.float32)

    def step(self, action, t):
        """Step environment forward."""
        self.step_count += 1
        
        # Fighter dynamics
        a_cmd = self._action_to_accel(action)
        a_total = a_cmd + CFG.GRAVITY
        
        self.fighter_vel += a_total * self.dt
        self.fighter_pos += self.fighter_vel * self.dt
        
        # Maintain cruise speed
        speed = np.linalg.norm(self.fighter_vel)
        if speed > 1e-3:
            self.fighter_vel *= CFG.TARG_INITIAL_SPEED / speed
        
        terminated = False
        
        # Ground collision
        if self.fighter_pos[2] < 0:
            self.fighter_pos[2] = 0
            terminated = True
        
        # === HARD ALTITUDE FLOOR - Terminate if too low ===
        # Real fighters don't dive to treetop level to evade missiles
        if self.fighter_pos[2] < 5000.0:
            terminated = True  # Episode ends - counts as failure
        
        # === Missile dynamics (simplified but realistic) ===
        t_flight = self.step_count * self.dt
        
        # Mass & thrust
        total_burn = CFG.MISS_BOOST_TIME + CFG.MISS_SUSTAIN_TIME
        if t_flight >= total_burn:
            mass = CFG.MISS_MASS_BURNOUT
            thrust = 0.0
        elif t_flight < CFG.MISS_BOOST_TIME:
            frac = t_flight / total_burn
            mass = CFG.MISS_MASS_INITIAL - (CFG.MISS_MASS_INITIAL - CFG.MISS_MASS_BURNOUT) * frac
            thrust = CFG.MISS_BOOST_THRUST
        else:
            frac = t_flight / total_burn
            mass = CFG.MISS_MASS_INITIAL - (CFG.MISS_MASS_INITIAL - CFG.MISS_MASS_BURNOUT) * frac
            thrust = CFG.MISS_SUSTAIN_THRUST
        
        # Geometry
        r = self.fighter_pos - self.missile_pos
        R = max(np.linalg.norm(r), 1e-6)
        los = r / R
        
        # PN Guidance
        rel_vel = self.fighter_vel - self.missile_vel
        Vc = -np.dot(rel_vel, los)
        v_perp = rel_vel - np.dot(rel_vel, los) * los
        los_rate = v_perp / R
        
        a_pn = CFG.N_PN * max(Vc, 0) * los_rate
        
        # Target accel estimation (APN)
        if self.prev_fighter_vel is not None:
            a_tgt = (self.fighter_vel - self.prev_fighter_vel) / self.dt
            a_mag = np.linalg.norm(a_tgt)
            if a_mag > 20 * 9.81:
                a_tgt *= 20 * 9.81 / a_mag
            a_tgt_perp = a_tgt - np.dot(a_tgt, los) * los
            a_pn += 0.5 * CFG.N_PN * a_tgt_perp
        
        self.prev_fighter_vel = self.fighter_vel.copy()
        
        # G-limit
        a_mag = np.linalg.norm(a_pn)
        max_g = CFG.MISS_MAX_G * 9.81
        if a_mag > max_g:
            a_pn *= max_g / a_mag
        
        # Thrust along velocity
        m_speed = np.linalg.norm(self.missile_vel)
        thrust_dir = self.missile_vel / max(m_speed, 1e-6)
        a_thrust = (thrust / mass) * thrust_dir
        
        # Drag
        alt = max(0, self.missile_pos[2])
        mach = m_speed / ATMOS.speed_of_sound(alt)
        cd = sim.get_missile_cd(mach)
        drag = sim.calculate_drag(self.missile_vel, alt, cd, CFG.MISS_REF_AREA)
        a_drag = drag / mass
        
        # Integrate missile
        a_total_m = a_thrust + a_drag + a_pn + CFG.GRAVITY
        self.missile_vel += a_total_m * self.dt
        self.missile_pos += self.missile_vel * self.dt
        
        if self.missile_pos[2] < 0:
            self.missile_pos[2] = 0
        
        # === REWARD FUNCTION (Anti-Exploit Design) ===
        reward = 0.0
        dist_km = R / 1000
        fighter_speed = np.linalg.norm(self.fighter_vel)
        alt = self.fighter_pos[2]

        # =====================================================================
        # 1. ALTITUDE COMPLIANCE (Primary constraint - ZERO TOLERANCE)
        # =====================================================================
        alt_min, alt_max = self.alt_band  # 9000, 13000
        
        # Reward for staying in band
        if alt_min <= alt <= alt_max:
            reward += 10.0 * self.dt  # Increased from 5.0
        
        # BRUTAL penalties for leaving the band - cubic scaling
        if alt < alt_min:
            deficit_km = (alt_min - alt) / 1000.0
            # Cubic penalty: violations grow FAST
            reward -= 200.0 * self.dt * (deficit_km ** 3)  # Was 50 * quadratic
            
            # Harsh tiered penalties - these stack!
            if alt < 8500:
                reward -= 50.0 * self.dt
            if alt < 8000:
                reward -= 100.0 * self.dt
            if alt < 7000:
                reward -= 300.0 * self.dt   # Was 100
            if alt < 6000:
                reward -= 500.0 * self.dt   # New tier
            if alt < 5000:
                reward -= 800.0 * self.dt   # Was 200
            if alt < 4000:
                reward -= 1000.0 * self.dt  # Was 300
        
        if alt > alt_max:
            excess_km = (alt - alt_max) / 1000.0
            reward -= 200.0 * self.dt * (excess_km ** 3)
            if alt > 14000:
                reward -= 100.0 * self.dt
            if alt > 15000:
                reward -= 300.0 * self.dt

        # =====================================================================
        # 2. ANTI-DIVE SYSTEM (Detect and punish diving behavior)
        # =====================================================================
        vert_vel = self.fighter_vel[2]  # Negative = diving
        
        # Punish ANY sustained diving
        if vert_vel < -20:  # Descending more than 20 m/s
            dive_rate = abs(vert_vel)
            reward -= 30.0 * self.dt * (dive_rate / 50.0) ** 2
        
        # Extra punishment for aggressive dives
        if vert_vel < -50:
            reward -= 80.0 * self.dt
        if vert_vel < -100:
            reward -= 200.0 * self.dt
        if vert_vel < -150:
            reward -= 400.0 * self.dt  # Insane dive rate
        
        # Punish diving when already below band
        if alt < alt_min and vert_vel < 0:
            # You're low AND still going down?! Massive penalty
            reward -= 100.0 * self.dt * (abs(vert_vel) / 30.0)

        # =====================================================================
        # 3. FLIGHT ATTITUDE CONSTRAINTS (No unrealistic flying)
        # =====================================================================
        vert_speed = abs(self.fighter_vel[2])
        
        # Vertical rate limits
        if vert_speed > 50:  # Tighter limit
            excess = (vert_speed - 50) / 30.0
            reward -= 40.0 * self.dt * (excess ** 2)
        
        if vert_speed > 100:
            reward -= 100.0 * self.dt
        
        if vert_speed > 150:
            reward -= 200.0 * self.dt
        
        # Flight path angle (pitch) limits
        if fighter_speed > 1e-3:
            fpa_rad = math.asin(np.clip(self.fighter_vel[2] / fighter_speed, -1, 1))
            fpa_deg = math.degrees(fpa_rad)  # Keep sign for dive detection
            
            # Penalize steep DIVES more than climbs
            if fpa_deg < -15:  # Diving
                excess = abs(fpa_deg) - 15
                reward -= 30.0 * self.dt * (excess / 10.0) ** 2
            
            if fpa_deg < -30:
                reward -= 100.0 * self.dt
            
            if fpa_deg < -45:
                reward -= 300.0 * self.dt
            
            # Climbs are less bad but still limited
            if fpa_deg > 25:
                excess = fpa_deg - 25
                reward -= 20.0 * self.dt * (excess / 15.0) ** 2

        # =====================================================================
        # 4. HORIZONTAL MANEUVERING REWARD (What we actually want)
        # =====================================================================
        a_horizontal = np.array([a_cmd[0], a_cmd[1], 0.0])
        a_horiz_mag = np.linalg.norm(a_horizontal)
        a_vert_mag = abs(a_cmd[2])
        
        # Only reward maneuvering if at proper altitude
        if alt_min <= alt <= alt_max:
            if dist_km < 15:
                horiz_g = a_horiz_mag / 9.81
                reward += horiz_g * 3.0  # Increased horizontal reward
                
                # Penalize vertical-dominated maneuvers
                if a_vert_mag > a_horiz_mag * 0.5 and a_vert_mag > 2 * 9.81:
                    reward -= 10.0 * self.dt

        # =====================================================================
        # 5. TACTICAL REWARDS (Only when compliant)
        # =====================================================================
        # Survival reward ONLY if flying properly
        if alt_min <= alt <= alt_max and abs(vert_vel) < 50:
            reward += 2.0 * self.dt  # Increased from 1.0
        
        # Distance rewards (reduced - don't incentivize cheese)
        if alt_min <= alt <= alt_max:  # Must be in band to get these
            if dist_km > 15:
                reward += 2.0
            elif dist_km > 10:
                reward += 1.0
            elif dist_km > 5:
                reward += 0.3
        
        # Close range danger (always applies)
        if dist_km < 5:
            reward -= 3.0 * (5 - dist_km) / 5
        
        # Beaming reward - only when flying properly
        if dist_km < 15 and m_speed > 1 and alt_min <= alt <= alt_max:
            fwd = self.fighter_vel / max(fighter_speed, 1e-3)
            cos_aspect = abs(np.dot(fwd, los))
            beam_quality = 1.0 - cos_aspect
            reward += beam_quality * 2.0

        # =====================================================================
        # 6. TERMINAL CONDITIONS
        # =====================================================================
        # Missile kill
        if R < self.kill_radius:
            reward -= 200.0  # Reduced - diving should be WORSE than this
            terminated = True
        
        # RAISED termination floor - 5000m instead of 3000m
        if self.fighter_pos[2] < 5000.0:
            reward -= 1000.0  # MUCH worse than getting hit
            terminated = True
        
        # Hard ceiling too
        if self.fighter_pos[2] > 18000.0:
            reward -= 500.0
            terminated = True
        
        # Time limit with strict bonus requirements
        truncated = self.step_count >= self.max_steps
        if truncated:
            if alt_min <= alt <= alt_max and abs(vert_vel) < 30:
                reward += 100.0  # Good survival bonus
            elif alt_min - 500 <= alt <= alt_max + 500:
                reward += 20.0   # Marginal bonus
            # else: no bonus for cheesing
        
        done = terminated or truncated
        return self._get_obs(), float(reward), done, {"R": R}


# ===========================================================================
# Neural Networks
# ===========================================================================

class FighterPolicyNet(nn.Module):
    """Policy network matching main simulation."""
    def __init__(self, obs_dim=14, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 4)
        )
    
    def forward(self, x):
        return self.net(x)


class ValueNet(nn.Module):
    """Value function approximator."""
    def __init__(self, obs_dim=14, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


# ===========================================================================
# PPO Training
# ===========================================================================

class PPOConfig:
    def __init__(self):
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_eps = 0.2
        self.entropy_coef = 0.01
        self.vf_coef = 0.5
        self.lr = 3e-4
        self.train_iters = 10
        self.minibatch_size = 4096
        self.steps_per_rollout = 16384
        self.total_steps = 3_000_000
        
        # Device selection
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        self.policy_ckpt = "fighter_policy.pt"
        self.value_ckpt = "fighter_value.pt"


def compute_gae(rewards, values, dones, gamma, lam):
    """Generalized Advantage Estimation."""
    T = len(rewards)
    advantages = torch.zeros(T, dtype=torch.float32, device=values.device)
    last_adv = 0.0
    
    for t in reversed(range(T)):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        last_adv = delta + gamma * lam * mask * last_adv
        advantages[t] = last_adv
    
    returns = advantages + values[:-1]
    return advantages, returns


def collect_bc_data(env, n_episodes=200):
    """Collect expert demonstrations for behavioral cloning."""
    obs_list, act_list = [], []
    t = 0.0
    
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = env.expert_action(t)
            obs_list.append(obs)
            act_list.append(action)
            obs, _, done, _ = env.step(action, t)
            t += env.dt
    
    print(f"[BC] Collected {len(obs_list)} samples from {n_episodes} episodes")
    return np.array(obs_list, dtype=np.float32), np.array(act_list, dtype=np.float32)


def train_bc(policy, env, device, n_episodes=200, epochs=30, batch_size=4096):
    """Behavioral cloning pretraining."""
    policy.to(device).train()
    
    obs, act = collect_bc_data(env, n_episodes)
    obs_t = torch.from_numpy(obs).to(device)
    act_t = torch.from_numpy(act).to(device)
    
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    n = len(obs)
    
    for epoch in range(epochs):
        idx = np.random.permutation(n)
        total_loss = 0.0
        
        for start in range(0, n, batch_size):
            batch = idx[start:start + batch_size]
            pred = policy(obs_t[batch])
            loss = nn.functional.mse_loss(pred, act_t[batch])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        
        print(f"[BC] Epoch {epoch+1}/{epochs}, Loss: {total_loss/n:.4f}")
    
    return policy


def train_ppo(run_bc=True):
    """Main PPO training loop."""
    cfg = PPOConfig()
    print(f"Using device: {cfg.device}")
    
    env = FighterEnv(dt=0.02, max_time=40.0, seed=42)
    
    policy = FighterPolicyNet().to(cfg.device)
    value_net = ValueNet().to(cfg.device)
    
    # Optional BC warmstart
    if run_bc:
        print("\n=== Behavioral Cloning Pretraining ===")
        train_bc(policy, env, cfg.device)
    
    # PPO setup
    log_std = nn.Parameter(torch.zeros(4, device=cfg.device))
    params = list(policy.parameters()) + list(value_net.parameters()) + [log_std]
    optimizer = optim.Adam(params, lr=cfg.lr)
    
    # Tracking
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    returns_history, survival_history = [], []
    
    global_step = 0
    ep_returns, ep_lengths = [0.0], [0]
    
    obs = env.reset()
    obs_t = torch.from_numpy(obs).unsqueeze(0).to(cfg.device)
    t = 0.0
    
    print("\n=== PPO Training ===")
    
    while global_step < cfg.total_steps:
        # Collect rollout
        obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], [], [], [], []
        
        for _ in range(cfg.steps_per_rollout):
            with torch.no_grad():
                mu = policy(obs_t)
                std = log_std.exp().expand_as(mu)
                dist = torch.distributions.Normal(mu, std)
                action_t = dist.sample()
                logp = dist.log_prob(action_t).sum(-1)
                val = value_net(obs_t)
            
            action = action_t.cpu().numpy()[0]
            next_obs, reward, done, info = env.step(action, t)
            t += env.dt
            
            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(logp.cpu().numpy()[0])
            rew_buf.append(reward)
            done_buf.append(done)
            val_buf.append(val.cpu().numpy()[0])
            
            global_step += 1
            ep_returns[-1] += reward
            ep_lengths[-1] += 1
            
            if done:
                obs = env.reset()
                ep_returns.append(0.0)
                ep_lengths.append(0)
            else:
                obs = next_obs
            
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(cfg.device)
        
        # Bootstrap
        with torch.no_grad():
            val_buf.append(value_net(obs_t).cpu().numpy()[0])
        
        # To tensors
        obs_t_batch = torch.from_numpy(np.array(obs_buf, dtype=np.float32)).to(cfg.device)
        act_t_batch = torch.from_numpy(np.array(act_buf, dtype=np.float32)).to(cfg.device)
        logp_old = torch.from_numpy(np.array(logp_buf, dtype=np.float32)).to(cfg.device)
        rew_t = torch.from_numpy(np.array(rew_buf, dtype=np.float32)).to(cfg.device)
        done_t = torch.from_numpy(np.array(done_buf)).to(cfg.device)
        val_t = torch.from_numpy(np.array(val_buf, dtype=np.float32)).to(cfg.device)
        
        # GAE
        adv, ret = compute_gae(rew_t, val_t, done_t, cfg.gamma, cfg.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        # PPO updates
        n = len(obs_buf)
        for _ in range(cfg.train_iters):
            idx = np.random.permutation(n)
            for start in range(0, n, cfg.minibatch_size):
                batch = idx[start:start + cfg.minibatch_size]
                
                mu = policy(obs_t_batch[batch])
                std = log_std.exp().expand_as(mu)
                dist = torch.distributions.Normal(mu, std)
                
                logp = dist.log_prob(act_t_batch[batch]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                v_pred = value_net(obs_t_batch[batch])
                
                ratio = torch.exp(logp - logp_old[batch])
                surr1 = ratio * adv[batch]
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv[batch]
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (ret[batch] - v_pred).pow(2).mean()
                loss = actor_loss + cfg.vf_coef * critic_loss - cfg.entropy_coef * entropy
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Logging
        recent_returns = ep_returns[-20:-1] if len(ep_returns) > 1 else [0]
        recent_lengths = ep_lengths[-20:-1] if len(ep_lengths) > 1 else [0]
        mean_ret = np.mean(recent_returns)
        mean_len = np.mean(recent_lengths)
        
        # Survival = reached time limit (not killed or crashed)
        survival_rate = np.mean([l >= env.max_steps * 0.95 for l in recent_lengths])
        
        # Proper survival = survived AND stayed in altitude band (no cheesing)
        # We can't track this perfectly here, but episode length near max is a proxy
        
        returns_history.append(mean_ret)
        survival_history.append(survival_rate * 100)
        
        print(f"[PPO] Step {global_step:,}/{cfg.total_steps:,} | "
              f"Return: {mean_ret:.1f} | Length: {mean_len:.0f} | Survival: {survival_rate*100:.0f}%")
        
        # Plot
        ax1.clear()
        ax1.plot(returns_history)
        ax1.set_xlabel("Update")
        ax1.set_ylabel("Mean Return")
        ax1.set_title("Training Progress")
        ax1.grid(True, alpha=0.3)
        
        ax2.clear()
        ax2.plot(survival_history)
        ax2.set_xlabel("Update")
        ax2.set_ylabel("Survival Rate (%)")
        ax2.set_title("Evasion Success")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.01)
        
        # Save checkpoints
        torch.save(policy.state_dict(), cfg.policy_ckpt)
        torch.save(value_net.state_dict(), cfg.value_ckpt)
    
    plt.ioff()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()
    
    print(f"\n=== Training Complete ===")
    print(f"Policy saved to: {cfg.policy_ckpt}")
    print(f"Value net saved to: {cfg.value_ckpt}")


if __name__ == "__main__":
    train_ppo(run_bc=True)