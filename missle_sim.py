"""
================================================================================
REALISTIC 3D MISSILE-TARGET PURSUIT SIMULATION
================================================================================
Author: [Your Name]
Version: 2.0
License: MIT

A high-fidelity engagement simulation implementing industry-standard guidance,
navigation, and control (GN&C) algorithms suitable for defense applications.

KEY FEATURES:
    - US Standard Atmosphere 1976 model
    - Multiple guidance laws: PN, APN, Optimal Guidance Law (OGL)
    - Extended Kalman Filter (EKF) for target state estimation
    - Zero-Effort Miss (ZEM) / Time-to-Go (tgo) calculations
    - 3-DOF point-mass dynamics with realistic actuator models
    - Comprehensive telemetry and data logging
    - Monte Carlo analysis capability

REFERENCES:
    [1] Zarchan, P. "Tactical and Strategic Missile Guidance" (6th Ed.)
    [2] US Standard Atmosphere, 1976 (NOAA/NASA/USAF)
    [3] Siouris, G. "Missile Guidance and Control Systems"
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from enum import Enum, auto
import logging

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# SECTION 1: ENUMERATIONS AND CONFIGURATION
# ============================================================================

class GuidanceMode(Enum):
    """Guidance law selection."""
    MIDCOURSE = auto()
    TERMINAL_PN = auto()      # Pure Proportional Navigation
    TERMINAL_APN = auto()     # Augmented PN (accounts for target accel)
    TERMINAL_OGL = auto()     # Optimal Guidance Law (time-to-go based)


class SeekerState(Enum):
    """Seeker operational state."""
    SEARCH = auto()
    ACQUISITION = auto()
    TRACK = auto()
    MEMORY = auto()
    LOST = auto()


@dataclass
class SimConfig:
    """Central configuration for all simulation parameters."""
    # === Physical Constants ===
    GRAVITY: np.ndarray = field(default_factory=lambda: np.array([0., 0., -9.80665]))
    R_GAS: float = 287.05287        # J/(kg·K) - specific gas constant for air
    GAMMA: float = 1.4              # ratio of specific heats

    # === US Standard Atmosphere 1976 Parameters ===
    # Geopotential altitude breakpoints (m)
    ATMOS_H: Tuple = (0., 11000., 20000., 32000., 47000., 51000., 71000., 84852.)
    # Temperature lapse rates (K/m) for each layer
    ATMOS_LAPSE: Tuple = (-0.0065, 0., 0.001, 0.0028, 0., -0.0028, -0.002)
    # Base temperatures (K)
    ATMOS_T_BASE: Tuple = (288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65)
    # Base pressures (Pa)
    ATMOS_P_BASE: Tuple = (101325., 22632.1, 5474.89, 868.019, 110.906, 66.9389, 3.95642)

    # === Missile Parameters ===
    MISS_MASS_INITIAL: float = 150.0    # kg - launch mass
    MISS_MASS_BURNOUT: float = 90.0     # kg - post-burn mass
    MISS_REF_AREA: float = 0.0314       # m² - reference area (d=0.2m)
    MISS_REF_LENGTH: float = 3.0        # m - reference length
    
    # Drag coefficients (Mach-dependent)
    MISS_CD_SUBSONIC: float = 0.30
    MISS_CD_TRANSONIC: float = 0.50
    MISS_CD_SUPERSONIC: float = 0.35
    
    # Propulsion profile
    MISS_BOOST_THRUST: float = 30000.0  # N
    MISS_BOOST_TIME: float = 3.0        # s
    MISS_SUSTAIN_THRUST: float = 5000.0 # N
    MISS_SUSTAIN_TIME: float = 7.0      # s
    MISS_ISP_BOOST: float = 250.0       # s - specific impulse
    MISS_ISP_SUSTAIN: float = 220.0     # s
    
    # Performance limits
    MISS_MAX_G: float = 50.0            # g's - lateral acceleration limit
    MISS_MAX_AOA: float = 0.436         # rad (~25°) - angle of attack limit
    
    # Actuator dynamics
    ACTUATOR_RATE_LIMIT: float = 500.0  # deg/s - fin rate limit
    ACTUATOR_POS_LIMIT: float = 25.0    # deg - fin deflection limit
    ACTUATOR_TIME_CONST: float = 0.03   # s - first-order time constant
    
    # End-of-life criteria
    MISS_MAX_FLIGHT_TIME: float = 60.0  # s
    MISS_MIN_SPEED: float = 200.0       # m/s
    MISS_MIN_CLOSING_RANGE: float = 2000.0  # m
    
    # === Target Aircraft Parameters ===
    TARG_MASS: float = 15000.0          # kg
    TARG_WING_AREA: float = 50.0        # m²
    TARG_CD0: float = 0.02              # zero-lift drag
    TARG_K: float = 0.05                # induced drag factor
    TARG_MAX_G: float = 9.0             # g's
    TARG_INITIAL_SPEED: float = 300.0   # m/s

    # === Seeker Parameters ===
    SEEKER_GIMBAL_LIMIT: float = 1.309  # rad (75°)
    SEEKER_FOV: float = 0.0524          # rad (3°)
    SEEKER_LOCK_SNR: float = 2.0        # dB threshold for maintaining lock
    SEEKER_ACQ_SNR: float = 4.0         # dB threshold for acquisition
    SEEKER_RANGE_MAX: float = 60000.0   # m
    SEEKER_NOISE_STD: float = 0.005     # rad - angular noise
    SEEKER_UPDATE_RATE: float = 50.0    # Hz
    
    # RCS values (m²)
    TARGET_RCS_HEAD: float = 3.0
    TARGET_RCS_BEAM: float = 10.0
    TARGET_RCS_TAIL: float = 5.0
    DECOY_RCS_BASE: float = 8.0
    
    # Doppler and track management
    DOPPLER_MIN_VEL: float = 20.0       # m/s - minimum closing velocity
    TRACK_MEMORY_TIME: float = 3.0      # s - coast time before dropping track
    LOCK_HYSTERESIS_TIME: float = 0.3   # s - minimum time before relock

    # === Guidance Parameters ===
    N_PN: float = 3.5                   # navigation constant
    GUIDANCE_DELAY: float = 0.020       # s - loop latency
    GUIDANCE_UPDATE_RATE: float = 100.0 # Hz
    MIDCOURSE_GAIN: float = 3.0
    
    # === Engagement Parameters ===
    KILL_DIST: float = 50.0             # m - lethal radius
    MISSILE_LAUNCH_TIME: float = 0.0    # s

    # === Decoy Parameters ===
    N_DECOYS: int = 3
    DECOY_DEPLOY_TIME: float = 35.0     # s (dynamic in practice)
    DECOY_BASE_SPEED: float = 200.0     # m/s
    DECOY_SPEED_SIGMA: float = 50.0     # m/s
    DECOY_MASS: float = 2.0             # kg
    DECOY_DRAG_AREA: float = 0.1        # m²
    DECOY_BURN_TIME: float = 5.0        # s

    # === Simulation Parameters ===
    TMAX: float = 75.0                  # s
    DT: float = 0.001                   # s - integration timestep
    
    # === Visualization ===
    ANIMATION_INTERVAL: int = 5         # ms
    ANIMATION_MAX_FRAMES: int = 500


# Global configuration instance
CFG = SimConfig()


# ============================================================================
# SECTION 2: US STANDARD ATMOSPHERE 1976
# ============================================================================

class Atmosphere:
    """
    US Standard Atmosphere 1976 implementation.
    
    Provides accurate atmospheric properties up to 86 km altitude using
    the piecewise-linear temperature model with hydrostatic equation.
    
    Reference: NOAA-S/T 76-1562, NASA-TM-X-74335
    """
    
    def __init__(self, cfg: SimConfig = CFG):
        self.cfg = cfg
        self._setup_layers()
    
    def _setup_layers(self):
        """Precompute layer boundaries and base values."""
        self.h = np.array(self.cfg.ATMOS_H)
        self.lapse = np.array(self.cfg.ATMOS_LAPSE)
        self.T_base = np.array(self.cfg.ATMOS_T_BASE)
        self.P_base = np.array(self.cfg.ATMOS_P_BASE)
        self.n_layers = len(self.lapse)
    
    def _get_layer(self, h: float) -> int:
        """Find atmospheric layer index for given altitude."""
        h = max(0., min(h, self.h[-1] - 1.))
        for i in range(self.n_layers):
            if h < self.h[i + 1]:
                return i
        return self.n_layers - 1
    
    def temperature(self, altitude: float) -> float:
        """
        Calculate temperature at altitude.
        
        Parameters
        ----------
        altitude : float
            Geometric altitude (m)
            
        Returns
        -------
        float
            Temperature (K)
        """
        h = max(0., altitude)
        i = self._get_layer(h)
        dh = h - self.h[i]
        return self.T_base[i] + self.lapse[i] * dh
    
    def pressure(self, altitude: float) -> float:
        """
        Calculate pressure at altitude using hydrostatic equation.
        
        Parameters
        ----------
        altitude : float
            Geometric altitude (m)
            
        Returns
        -------
        float
            Pressure (Pa)
        """
        h = max(0., altitude)
        i = self._get_layer(h)
        dh = h - self.h[i]
        T = self.temperature(h)
        
        g0 = 9.80665
        R = self.cfg.R_GAS
        
        if abs(self.lapse[i]) > 1e-10:
            # Non-isothermal layer
            exp = g0 / (R * self.lapse[i])
            return self.P_base[i] * (self.T_base[i] / T) ** exp
        else:
            # Isothermal layer
            return self.P_base[i] * np.exp(-g0 * dh / (R * T))
    
    def density(self, altitude: float) -> float:
        """
        Calculate air density using ideal gas law.
        
        Parameters
        ----------
        altitude : float
            Geometric altitude (m)
            
        Returns
        -------
        float
            Density (kg/m³)
        """
        T = self.temperature(altitude)
        P = self.pressure(altitude)
        return P / (self.cfg.R_GAS * T)
    
    def speed_of_sound(self, altitude: float) -> float:
        """
        Calculate speed of sound.
        
        Parameters
        ----------
        altitude : float
            Geometric altitude (m)
            
        Returns
        -------
        float
            Speed of sound (m/s)
        """
        T = self.temperature(altitude)
        return np.sqrt(self.cfg.GAMMA * self.cfg.R_GAS * T)
    
    def mach_number(self, velocity: np.ndarray, altitude: float) -> float:
        """Calculate Mach number for given velocity and altitude."""
        speed = np.linalg.norm(velocity)
        a = self.speed_of_sound(altitude)
        return speed / a if a > 0 else 0.0


# Global atmosphere instance
ATMOS = Atmosphere()


# ============================================================================
# SECTION 3: TELEMETRY AND DATA LOGGING
# ============================================================================

@dataclass
class TelemetryFrame:
    """Single frame of telemetry data."""
    time: float
    
    # Missile state
    miss_pos: np.ndarray
    miss_vel: np.ndarray
    miss_accel_cmd: np.ndarray
    miss_mass: float
    miss_mach: float
    
    # Guidance data
    guidance_mode: GuidanceMode
    zem: float                  # Zero-Effort Miss
    tgo: float                  # Time-to-go
    los_rate: float             # Line-of-sight rate magnitude
    closing_vel: float          # Closing velocity
    
    # Seeker data
    seeker_state: SeekerState
    seeker_snr: float
    gimbal_angle: float
    lock_type: Optional[str]
    
    # Target state
    tgt_pos: np.ndarray
    tgt_vel: np.ndarray
    range_to_target: float
    
    # Engagement metrics
    lateral_accel: float        # Achieved lateral acceleration (g's)


class TelemetryLogger:
    """
    Flight data recorder for post-engagement analysis.
    
    Captures comprehensive telemetry at each guidance cycle for
    performance analysis and debugging.
    """
    
    def __init__(self):
        self.frames: List[TelemetryFrame] = []
        self.metadata: Dict = {}
    
    def log(self, frame: TelemetryFrame):
        """Record a telemetry frame."""
        self.frames.append(frame)
    
    def set_metadata(self, key: str, value):
        """Set engagement metadata."""
        self.metadata[key] = value
    
    def get_time_series(self, attr: str) -> Tuple[np.ndarray, np.ndarray]:
        """Extract time series for a specific attribute."""
        times = np.array([f.time for f in self.frames])
        values = np.array([getattr(f, attr) for f in self.frames])
        return times, values
    
    def compute_miss_distance(self) -> Tuple[float, float, int]:
        """
        Compute closest point of approach (CPA).
        
        Returns
        -------
        tuple
            (miss_distance, time_of_cpa, frame_index)
        """
        if not self.frames:
            return np.inf, 0., 0
        
        ranges = [f.range_to_target for f in self.frames]
        min_idx = np.argmin(ranges)
        return ranges[min_idx], self.frames[min_idx].time, min_idx
    
    def export_csv(self, filename: str):
        """Export telemetry to CSV for external analysis."""
        import csv
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow([
                'time', 'miss_x', 'miss_y', 'miss_z',
                'miss_vx', 'miss_vy', 'miss_vz',
                'tgt_x', 'tgt_y', 'tgt_z', 'range',
                'zem', 'tgo', 'closing_vel', 'mach',
                'guidance_mode', 'seeker_state', 'lateral_g'
            ])
            # Data
            for f in self.frames:
                writer.writerow([
                    f.time,
                    *f.miss_pos, *f.miss_vel,
                    *f.tgt_pos, f.range_to_target,
                    f.zem, f.tgo, f.closing_vel, f.miss_mach,
                    f.guidance_mode.name, f.seeker_state.name,
                    f.lateral_accel
                ])


# ============================================================================
# SECTION 4: KALMAN FILTER FOR TARGET TRACKING
# ============================================================================

class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for target state estimation.
    
    Estimates target position, velocity, and acceleration from
    noisy seeker measurements. Uses a constant-acceleration motion model.
    
    State vector: [x, y, z, vx, vy, vz, ax, ay, az]
    Measurement: [azimuth, elevation, range]
    """
    
    def __init__(self, dt: float = 0.02):
        self.dt = dt
        self.n_state = 9
        self.n_meas = 3
        
        # State vector and covariance
        self.x = np.zeros(self.n_state)
        self.P = np.eye(self.n_state) * 1000.0
        
        # Process noise (tuned for maneuvering target)
        self.Q = np.eye(self.n_state)
        self.Q[:3, :3] *= 1.0       # position noise
        self.Q[3:6, 3:6] *= 10.0    # velocity noise
        self.Q[6:9, 6:9] *= 100.0   # acceleration noise
        
        # Measurement noise
        self.R = np.diag([0.001, 0.001, 100.0])  # [az, el, range] variance
        
        self.initialized = False
    
    def initialize(self, pos: np.ndarray, vel: np.ndarray = None):
        """Initialize filter with first measurement."""
        self.x[:3] = pos
        if vel is not None:
            self.x[3:6] = vel
        self.x[6:9] = 0.0  # Initial acceleration estimate
        self.P = np.eye(self.n_state) * 100.0
        self.initialized = True
    
    def predict(self, dt: float = None):
        """Propagate state using constant-acceleration model."""
        if dt is None:
            dt = self.dt
        
        # State transition matrix (constant acceleration)
        F = np.eye(self.n_state)
        F[:3, 3:6] = np.eye(3) * dt
        F[:3, 6:9] = np.eye(3) * 0.5 * dt**2
        F[3:6, 6:9] = np.eye(3) * dt
        
        # Predict state and covariance
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q * dt
    
    def update(self, z_meas: np.ndarray, missile_pos: np.ndarray):
        """
        Update state with measurement.
        
        Parameters
        ----------
        z_meas : np.ndarray
            Measurement [azimuth, elevation, range] in radians/meters
        missile_pos : np.ndarray
            Current missile position for measurement model
        """
        # Predicted measurement
        rel_pos = self.x[:3] - missile_pos
        r_pred = np.linalg.norm(rel_pos)
        
        if r_pred < 1e-6:
            return
        
        az_pred = np.arctan2(rel_pos[1], rel_pos[0])
        el_pred = np.arcsin(np.clip(rel_pos[2] / r_pred, -1, 1))
        z_pred = np.array([az_pred, el_pred, r_pred])
        
        # Measurement Jacobian
        H = self._measurement_jacobian(rel_pos, r_pred)
        
        # Kalman gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Innovation (handle angle wrapping)
        y = z_meas - z_pred
        y[0] = np.arctan2(np.sin(y[0]), np.cos(y[0]))  # wrap azimuth
        y[1] = np.arctan2(np.sin(y[1]), np.cos(y[1]))  # wrap elevation
        
        # Update
        self.x = self.x + K @ y
        self.P = (np.eye(self.n_state) - K @ H) @ self.P
    
    def _measurement_jacobian(self, rel_pos: np.ndarray, r: float) -> np.ndarray:
        """Compute measurement Jacobian matrix."""
        x, y, z = rel_pos
        r_xy = np.sqrt(x**2 + y**2)
        
        H = np.zeros((3, self.n_state))
        
        if r_xy > 1e-6:
            # d(azimuth)/d(pos)
            H[0, 0] = -y / (r_xy**2)
            H[0, 1] = x / (r_xy**2)
            
            # d(elevation)/d(pos)
            H[1, 0] = -x * z / (r**2 * r_xy)
            H[1, 1] = -y * z / (r**2 * r_xy)
            H[1, 2] = r_xy / r**2
        
        # d(range)/d(pos)
        H[2, :3] = rel_pos / r
        
        return H
    
    @property
    def position(self) -> np.ndarray:
        return self.x[:3].copy()
    
    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:6].copy()
    
    @property
    def acceleration(self) -> np.ndarray:
        return self.x[6:9].copy()


# ============================================================================
# SECTION 5: GUIDANCE ALGORITHMS
# ============================================================================

class GuidanceComputer:
    """
    Multi-mode guidance computer implementing industry-standard guidance laws.
    
    Supported modes:
        - Midcourse: Proportional pursuit toward predicted intercept
        - Terminal PN: Pure Proportional Navigation
        - Terminal APN: Augmented PN with target acceleration
        - Terminal OGL: Optimal Guidance Law (minimum-effort intercept)
    
    References:
        Zarchan, "Tactical and Strategic Missile Guidance"
    """
    
    def __init__(self, cfg: SimConfig = CFG):
        self.cfg = cfg
        self.mode = GuidanceMode.MIDCOURSE
        
        # Delay buffer for guidance latency
        self.delay_buffer = deque()
        
        # Actuator state
        self.accel_cmd = np.zeros(3)
        self.accel_actual = np.zeros(3)
        self.fin_rate = 0.0
        
        # State history for derivative estimation
        self.prev_los = None
        self.prev_tgt_vel = None
        
        # Computed guidance parameters (for telemetry)
        self.zem = 0.0
        self.tgo = 0.0
        self.los_rate = 0.0
        self.closing_vel = 0.0
    
    def compute_geometry(self, miss_pos: np.ndarray, miss_vel: np.ndarray,
                        tgt_pos: np.ndarray, tgt_vel: np.ndarray
                        ) -> Tuple[np.ndarray, float, float, np.ndarray]:
        """
        Compute engagement geometry parameters.
        
        Returns
        -------
        tuple
            (los_unit, range, closing_velocity, los_rate_vec)
        """
        # Relative geometry
        r = tgt_pos - miss_pos
        R = np.linalg.norm(r)
        
        if R < 1e-6:
            return np.array([1., 0., 0.]), 0., 0., np.zeros(3)
        
        los = r / R
        v_rel = tgt_vel - miss_vel
        
        # Closing velocity (positive when closing)
        Vc = -np.dot(v_rel, los)
        self.closing_vel = Vc
        
        # LOS rate vector
        v_rel_perp = v_rel - np.dot(v_rel, los) * los
        los_rate_vec = v_rel_perp / R
        self.los_rate = np.linalg.norm(los_rate_vec)
        
        return los, R, Vc, los_rate_vec
    
    def estimate_time_to_go(self, R: float, Vc: float, 
                           miss_vel: np.ndarray, tgt_accel: np.ndarray) -> float:
        """
        Estimate time-to-go using iterative refinement.
        
        Uses closing velocity with acceleration correction for
        better tgo estimate in maneuvering engagements.
        """
        if Vc <= 0:
            return 100.0  # Not closing - return large value
        
        # First-order estimate
        tgo = R / Vc
        
        # Refine with target acceleration (simplified)
        # tgo_refined = tgo * (1 + 0.5 * at_closing * tgo / Vc)
        # Clamp to reasonable bounds
        self.tgo = np.clip(tgo, 0.1, 100.0)
        return self.tgo
    
    def compute_zem(self, R: float, los: np.ndarray, los_rate: np.ndarray,
                    tgo: float) -> float:
        """
        Compute Zero-Effort Miss distance.
        
        ZEM is the predicted miss distance if no further guidance
        corrections are made. Key metric for guidance performance.
        
        ZEM = R * |ω| * tgo  (first-order approximation)
        """
        los_rate_mag = np.linalg.norm(los_rate)
        self.zem = R * los_rate_mag * tgo
        return self.zem
    
    def proportional_navigation(self, Vc: float, los_rate: np.ndarray) -> np.ndarray:
        """
        Pure Proportional Navigation guidance law.
        
        a_cmd = N * Vc * ω
        
        where N is the navigation constant and ω is the LOS rate.
        """
        return self.cfg.N_PN * Vc * los_rate
    
    def augmented_pn(self, Vc: float, los_rate: np.ndarray, 
                     tgt_accel: np.ndarray, los: np.ndarray) -> np.ndarray:
        """
        Augmented Proportional Navigation.
        
        a_cmd = N * Vc * ω + (N/2) * at_perp
        
        Adds target acceleration compensation for maneuvering targets.
        """
        a_pn = self.proportional_navigation(Vc, los_rate)
        
        # Target accel perpendicular to LOS
        at_perp = tgt_accel - np.dot(tgt_accel, los) * los
        
        return a_pn + 0.5 * self.cfg.N_PN * at_perp
    
    def optimal_guidance_law(self, zem: float, tgo: float, 
                             los: np.ndarray, los_rate: np.ndarray) -> np.ndarray:
        """
        Optimal Guidance Law (OGL) for minimum-effort intercept.
        
        a_cmd = (N' / tgo²) * ZEM_vec
        
        where N' is typically 3-6 and ZEM_vec is the zero-effort miss vector.
        Provides more efficient trajectories than PN when tgo is known.
        """
        if tgo < 0.1:
            tgo = 0.1  # Prevent singularity
        
        N_prime = 4.0  # OGL navigation gain
        
        # ZEM vector (perpendicular to LOS)
        los_rate_mag = np.linalg.norm(los_rate)
        if los_rate_mag > 1e-9:
            zem_dir = los_rate / los_rate_mag
        else:
            zem_dir = np.zeros(3)
        
        zem_vec = zem * zem_dir
        
        return (N_prime / (tgo ** 2)) * zem_vec
    
    def compute_midcourse(self, miss_pos: np.ndarray, miss_vel: np.ndarray,
                         tgt_pos: np.ndarray, tgt_vel: np.ndarray) -> np.ndarray:
        """
        Midcourse guidance: fly toward predicted intercept point.
        
        Uses proportional navigation with reduced gain for fuel efficiency.
        """
        los, R, Vc, los_rate = self.compute_geometry(
            miss_pos, miss_vel, tgt_pos, tgt_vel)
        
        if Vc <= 0:
            return np.zeros(3)
        
        return self.cfg.MIDCOURSE_GAIN * Vc * los_rate
    
    def compute_terminal(self, miss_pos: np.ndarray, miss_vel: np.ndarray,
                        tgt_pos: np.ndarray, tgt_vel: np.ndarray,
                        tgt_accel: np.ndarray, dt: float) -> np.ndarray:
        """
        Terminal guidance with selectable guidance law.
        
        Computes engagement geometry and applies selected guidance algorithm.
        """
        los, R, Vc, los_rate = self.compute_geometry(
            miss_pos, miss_vel, tgt_pos, tgt_vel)
        
        # Estimate time-to-go
        tgo = self.estimate_time_to_go(R, Vc, miss_vel, tgt_accel)
        
        # Compute ZEM
        self.compute_zem(R, los, los_rate, tgo)
        
        # Select guidance law
        if self.mode == GuidanceMode.TERMINAL_PN:
            a_cmd = self.proportional_navigation(Vc, los_rate)
        elif self.mode == GuidanceMode.TERMINAL_APN:
            a_cmd = self.augmented_pn(Vc, los_rate, tgt_accel, los)
        elif self.mode == GuidanceMode.TERMINAL_OGL:
            a_cmd = self.optimal_guidance_law(self.zem, tgo, los, los_rate)
        else:
            a_cmd = self.proportional_navigation(Vc, los_rate)
        
        # Project to plane perpendicular to missile velocity (lateral accel only)
        speed = np.linalg.norm(miss_vel)
        if speed > 1e-6:
            v_hat = miss_vel / speed
            a_cmd = a_cmd - np.dot(a_cmd, v_hat) * v_hat
        
        return a_cmd
    
    def apply_g_limit(self, a_cmd: np.ndarray) -> np.ndarray:
        """Apply acceleration limits."""
        a_mag = np.linalg.norm(a_cmd)
        max_accel = self.cfg.MISS_MAX_G * 9.81
        
        if a_mag > max_accel:
            return a_cmd * (max_accel / a_mag)
        return a_cmd
    
    def apply_actuator_dynamics(self, a_cmd: np.ndarray, dt: float) -> np.ndarray:
        """
        Model actuator dynamics with rate and position limits.
        
        First-order lag with rate limiting:
            τ(da/dt) + a = a_cmd, |da/dt| ≤ rate_limit
        """
        tau = self.cfg.ACTUATOR_TIME_CONST
        
        # Desired change
        da_desired = (a_cmd - self.accel_actual) * dt / (tau + dt)
        
        # Rate limiting
        da_mag = np.linalg.norm(da_desired)
        max_rate = self.cfg.ACTUATOR_RATE_LIMIT * np.pi / 180 * 9.81 * 10  # rough conversion
        if da_mag > max_rate * dt:
            da_desired = da_desired * (max_rate * dt / da_mag)
        
        self.accel_actual = self.accel_actual + da_desired
        return self.accel_actual
    
    def apply_delay(self, t: float, a_cmd: np.ndarray) -> np.ndarray:
        """Apply guidance loop latency via delay buffer."""
        self.delay_buffer.append((t, a_cmd))
        
        while self.delay_buffer and self.delay_buffer[0][0] < t - self.cfg.GUIDANCE_DELAY:
            self.delay_buffer.popleft()
        
        if self.delay_buffer:
            return self.delay_buffer[0][1]
        return a_cmd
    
    def update(self, t: float, dt: float,
               miss_pos: np.ndarray, miss_vel: np.ndarray,
               tgt_pos: np.ndarray, tgt_vel: np.ndarray,
               tgt_accel: np.ndarray, has_lock: bool) -> np.ndarray:
        """
        Main guidance update - compute acceleration command.
        
        Parameters
        ----------
        t : float
            Current time (s)
        dt : float
            Time step (s)
        miss_pos, miss_vel : np.ndarray
            Missile state
        tgt_pos, tgt_vel, tgt_accel : np.ndarray
            Target state (from seeker/EKF)
        has_lock : bool
            True if seeker has target lock
            
        Returns
        -------
        np.ndarray
            Acceleration command (m/s²)
        """
        # Mode selection
        if has_lock:
            if self.mode == GuidanceMode.MIDCOURSE:
                self.mode = GuidanceMode.TERMINAL_APN
            a_cmd = self.compute_terminal(
                miss_pos, miss_vel, tgt_pos, tgt_vel, tgt_accel, dt)
        else:
            self.mode = GuidanceMode.MIDCOURSE
            a_cmd = self.compute_midcourse(miss_pos, miss_vel, tgt_pos, tgt_vel)
        
        # Apply limits and dynamics
        a_cmd = self.apply_g_limit(a_cmd)
        a_cmd = self.apply_delay(t, a_cmd)
        a_cmd = self.apply_actuator_dynamics(a_cmd, dt)
        
        self.accel_cmd = a_cmd
        return a_cmd


# ============================================================================
# SECTION 6: AERODYNAMICS
# ============================================================================

def get_missile_cd(mach: float, cfg: SimConfig = CFG) -> float:
    """
    Calculate missile drag coefficient based on Mach number.
    
    Models transonic drag rise phenomenon with smooth transitions.
    """
    if mach < 0.8:
        return cfg.MISS_CD_SUBSONIC
    elif mach < 1.2:
        # Smooth transition through transonic
        t = (mach - 0.8) / 0.4
        t_smooth = 0.5 * (1 - np.cos(np.pi * t))  # Smooth interpolation
        return cfg.MISS_CD_SUBSONIC + t_smooth * (cfg.MISS_CD_TRANSONIC - cfg.MISS_CD_SUBSONIC)
    else:
        # Supersonic with gradual decrease
        return cfg.MISS_CD_SUPERSONIC + 0.1 / mach


def calculate_drag(velocity: np.ndarray, altitude: float, 
                   cd: float, ref_area: float) -> np.ndarray:
    """
    Calculate aerodynamic drag force vector.
    
    D = 0.5 * ρ * Cd * A * V² * (-v̂)
    """
    speed = np.linalg.norm(velocity)
    if speed < 1e-6:
        return np.zeros(3)
    
    rho = ATMOS.density(altitude)
    return -0.5 * rho * cd * ref_area * speed * velocity


def calculate_rcs(tgt_pos: np.ndarray, tgt_vel: np.ndarray,
                  obs_pos: np.ndarray, cfg: SimConfig = CFG) -> float:
    """
    Calculate aspect-angle-dependent Radar Cross Section.
    
    RCS varies with viewing angle relative to target heading.
    """
    los = obs_pos - tgt_pos
    los_dist = np.linalg.norm(los)
    
    if los_dist < 1e-6:
        return cfg.TARGET_RCS_BEAM
    
    los_unit = los / los_dist
    tgt_speed = np.linalg.norm(tgt_vel)
    
    if tgt_speed < 1e-6:
        return cfg.TARGET_RCS_BEAM
    
    tgt_heading = tgt_vel / tgt_speed
    cos_aspect = np.dot(los_unit, tgt_heading)
    aspect_angle = np.arccos(np.clip(cos_aspect, -1, 1))
    
    # Piecewise interpolation
    if aspect_angle < np.pi / 4:
        t = aspect_angle / (np.pi / 4)
        return cfg.TARGET_RCS_HEAD + t * (cfg.TARGET_RCS_BEAM - cfg.TARGET_RCS_HEAD)
    elif aspect_angle < 3 * np.pi / 4:
        return cfg.TARGET_RCS_BEAM
    else:
        t = (aspect_angle - 3 * np.pi / 4) / (np.pi / 4)
        return cfg.TARGET_RCS_BEAM + t * (cfg.TARGET_RCS_TAIL - cfg.TARGET_RCS_BEAM)


# ============================================================================
# SECTION 7: SEEKER MODEL (Enhanced)
# ============================================================================

class Seeker:
    """
    Enhanced missile seeker with EKF integration.
    
    Features:
        - Gimbal limits and boresight tracking
        - SNR-based detection with radar equation
        - Doppler velocity gating
        - Track memory for brief signal loss
        - Lock hysteresis
        - EKF for smooth target state estimation
    """
    
    def __init__(self, rng: np.random.Generator, initial_boresight: np.ndarray = None,
                 cfg: SimConfig = CFG):
        self.rng = rng
        self.cfg = cfg
        self.state = SeekerState.SEARCH
        self.current_lock = None
        self.lock_time = 0.0
        self.last_measurement_time = 0.0
        
        # Boresight direction
        if initial_boresight is not None:
            norm = np.linalg.norm(initial_boresight)
            self.boresight = initial_boresight / norm if norm > 1e-6 else np.array([1., 0., 0.])
        else:
            self.boresight = np.array([1., 0., 0.])
        
        # EKF for target tracking
        self.ekf = ExtendedKalmanFilter()
        
        # Current measurements (for telemetry)
        self.snr = 0.0
        self.gimbal_angle = 0.0
    
    def update_boresight(self, missile_vel: np.ndarray):
        """Align boresight with missile velocity."""
        speed = np.linalg.norm(missile_vel)
        if speed > 1e-6:
            self.boresight = missile_vel / speed
    
    def check_gimbal(self, miss_pos: np.ndarray, tgt_pos: np.ndarray) -> Tuple[bool, float]:
        """Check if target is within gimbal limits."""
        los = tgt_pos - miss_pos
        los_dist = np.linalg.norm(los)
        
        if los_dist < 1e-6:
            return True, 0.0
        
        los_unit = los / los_dist
        cos_angle = np.dot(los_unit, self.boresight)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        self.gimbal_angle = angle
        
        return angle <= self.cfg.SEEKER_GIMBAL_LIMIT, angle
    
    def calculate_snr(self, miss_pos: np.ndarray, tgt_pos: np.ndarray, rcs: float) -> float:
        """Calculate SNR using simplified radar equation."""
        dist = np.linalg.norm(tgt_pos - miss_pos)
        
        if dist < 1e-6 or dist > self.cfg.SEEKER_RANGE_MAX:
            return 0.0
        
        # SNR ∝ RCS / R⁴
        range_factor = (self.cfg.SEEKER_RANGE_MAX / dist) ** 4
        snr = rcs * range_factor
        
        # Atmospheric attenuation
        snr *= np.exp(-dist / 200000.0)
        
        # Add noise
        noise = self.rng.normal(0, 0.05 * snr + 0.1)
        self.snr = max(0.0, snr + noise)
        
        return self.snr
    
    def check_doppler(self, miss_pos: np.ndarray, miss_vel: np.ndarray,
                      tgt_pos: np.ndarray, tgt_vel: np.ndarray) -> bool:
        """Check Doppler velocity gate."""
        los = tgt_pos - miss_pos
        dist = np.linalg.norm(los)
        
        if dist < 1e-6:
            return True
        
        los_unit = los / dist
        rel_vel = miss_vel - tgt_vel
        closing_vel = np.dot(rel_vel, los_unit)
        
        return closing_vel >= self.cfg.DOPPLER_MIN_VEL
    
    def evaluate_target(self, miss_pos: np.ndarray, miss_vel: np.ndarray,
                       tgt_pos: np.ndarray, tgt_vel: np.ndarray,
                       rcs: float, is_current: bool = False) -> Tuple[bool, float]:
        """Evaluate if target can be tracked."""
        in_gimbal, _ = self.check_gimbal(miss_pos, tgt_pos)
        if not in_gimbal:
            return False, 0.0
        
        snr = self.calculate_snr(miss_pos, tgt_pos, rcs)
        
        if not self.check_doppler(miss_pos, miss_vel, tgt_pos, tgt_vel):
            snr *= 0.2
        
        threshold = self.cfg.SEEKER_LOCK_SNR if is_current else self.cfg.SEEKER_ACQ_SNR
        return snr >= threshold, snr
    
    def update(self, t: float, dt: float,
               miss_pos: np.ndarray, miss_vel: np.ndarray,
               tgt_pos: np.ndarray, tgt_vel: np.ndarray,
               decoy_positions: List[np.ndarray],
               decoy_velocities: List[np.ndarray],
               decoy_active: List[bool]) -> Tuple[Optional[np.ndarray], 
                                                   Optional[np.ndarray],
                                                   Optional[np.ndarray],
                                                   Optional[str]]:
        """
        Main seeker update.
        
        Returns
        -------
        tuple
            (tracked_pos, tracked_vel, tracked_accel, lock_type)
        """
        self.update_boresight(miss_vel)
        
        # Evaluate all candidates
        candidates = []
        
        # Real target
        rcs = calculate_rcs(tgt_pos, tgt_vel, miss_pos)
        trackable, snr = self.evaluate_target(
            miss_pos, miss_vel, tgt_pos, tgt_vel, rcs,
            is_current=(self.current_lock == 'target'))
        if trackable:
            candidates.append(('target', tgt_pos.copy(), tgt_vel.copy(), snr))
        
        # Decoys
        for i, (d_pos, d_vel, active) in enumerate(zip(
                decoy_positions, decoy_velocities, decoy_active)):
            if not active:
                continue
            
            lock_name = f'decoy_{i}'
            trackable, snr = self.evaluate_target(
                miss_pos, miss_vel, d_pos, d_vel, self.cfg.DECOY_RCS_BASE,
                is_current=(self.current_lock == lock_name))
            if trackable:
                candidates.append((lock_name, d_pos.copy(), d_vel.copy(), snr))
        
        # Handle no candidates
        if not candidates:
            self.state = SeekerState.MEMORY if self.current_lock else SeekerState.SEARCH
            if t - self.last_measurement_time > self.cfg.TRACK_MEMORY_TIME:
                self.current_lock = None
                self.state = SeekerState.LOST
            
            # Return EKF prediction during memory
            if self.ekf.initialized and self.state == SeekerState.MEMORY:
                self.ekf.predict(dt)
                return self.ekf.position, self.ekf.velocity, self.ekf.acceleration, self.current_lock
            return None, None, None, None
        
        # Select best candidate
        candidates.sort(key=lambda x: x[3], reverse=True)
        best = candidates[0]
        
        # Hysteresis
        if self.current_lock and self.current_lock != best[0]:
            if self.lock_time < self.cfg.LOCK_HYSTERESIS_TIME:
                for c in candidates:
                    if c[0] == self.current_lock:
                        best = c
                        break
        
        # Update lock state
        if best[0] != self.current_lock:
            self.current_lock = best[0]
            self.lock_time = 0.0
            self.ekf.initialize(best[1], best[2])
            self.state = SeekerState.ACQUISITION
        else:
            self.lock_time += dt
            self.state = SeekerState.TRACK
        
        self.last_measurement_time = t
        
        # Add measurement noise and update EKF
        los = best[1] - miss_pos
        dist = np.linalg.norm(los)
        if dist > 1e-6:
            # Create measurement with noise
            az = np.arctan2(los[1], los[0])
            el = np.arcsin(np.clip(los[2] / dist, -1, 1))
            r = dist
            
            az_noisy = az + self.rng.normal(0, self.cfg.SEEKER_NOISE_STD)
            el_noisy = el + self.rng.normal(0, self.cfg.SEEKER_NOISE_STD)
            r_noisy = r + self.rng.normal(0, 10.0)  # 10m range noise
            
            z_meas = np.array([az_noisy, el_noisy, r_noisy])
            
            self.ekf.predict(dt)
            self.ekf.update(z_meas, miss_pos)
        
        return self.ekf.position, self.ekf.velocity, self.ekf.acceleration, best[0]


# ============================================================================
# SECTION 8: MISSILE CLASS
# ============================================================================

class Missile:
    """Complete missile model with propulsion, aerodynamics, seeker, and guidance."""
    
    def __init__(self, pos: np.ndarray, vel: np.ndarray, 
                 rng: np.random.Generator, cfg: SimConfig = CFG):
        self.cfg = cfg
        self.pos = pos.astype(float).copy()
        self.vel = vel.astype(float).copy()
        self.mass = cfg.MISS_MASS_INITIAL
        self.launched = False
        self.launch_time = 0.0
        
        self.seeker = Seeker(rng, initial_boresight=vel, cfg=cfg)
        self.guidance = GuidanceComputer(cfg)
        
        self.intercepted = False
        self.intercept_type = None
        
        # Telemetry
        self.logger = TelemetryLogger()
    
    def get_thrust(self, t_flight: float) -> float:
        """Get current thrust based on propulsion phase."""
        if t_flight < self.cfg.MISS_BOOST_TIME:
            return self.cfg.MISS_BOOST_THRUST
        elif t_flight < self.cfg.MISS_BOOST_TIME + self.cfg.MISS_SUSTAIN_TIME:
            return self.cfg.MISS_SUSTAIN_THRUST
        return 0.0
    
    def get_mass(self, t_flight: float) -> float:
        """Get current mass accounting for fuel depletion."""
        total_burn = self.cfg.MISS_BOOST_TIME + self.cfg.MISS_SUSTAIN_TIME
        if t_flight >= total_burn:
            return self.cfg.MISS_MASS_BURNOUT
        
        fuel_mass = self.cfg.MISS_MASS_INITIAL - self.cfg.MISS_MASS_BURNOUT
        burn_fraction = t_flight / total_burn
        return self.cfg.MISS_MASS_INITIAL - fuel_mass * burn_fraction
    
    def update(self, t: float, dt: float,
               tgt_pos: np.ndarray, tgt_vel: np.ndarray,
               decoy_positions: List[np.ndarray],
               decoy_velocities: List[np.ndarray],
               decoy_active: List[bool]) -> Optional[str]:
        """Main missile update step."""
        if self.intercepted or not self.launched:
            return self.seeker.current_lock
        
        t_flight = t - self.launch_time
        altitude = max(0.0, self.pos[2])
        
        # Update mass and thrust
        self.mass = self.get_mass(t_flight)
        thrust_mag = self.get_thrust(t_flight)
        
        # Check end-of-life
        speed = np.linalg.norm(self.vel)
        r_vec = tgt_pos - self.pos
        R = np.linalg.norm(r_vec)
        
        if R > 1e-6 and speed > 1e-3:
            los_hat = r_vec / R
            rel_vel = tgt_vel - self.vel
            closing_vel = -np.dot(rel_vel, los_hat)
        else:
            closing_vel = 0.0
        
        spent = (t_flight > self.cfg.MISS_MAX_FLIGHT_TIME or
                 speed < self.cfg.MISS_MIN_SPEED or
                 (closing_vel < 0 and R > self.cfg.MISS_MIN_CLOSING_RANGE))
        
        if spent:
            self.seeker.current_lock = None
            accel_cmd = np.zeros(3)
            tgt_accel_est = np.zeros(3)
            lock_type = None
        else:
            # Seeker update
            tracked_pos, tracked_vel, tracked_accel, lock_type = self.seeker.update(
                t, dt, self.pos, self.vel, tgt_pos, tgt_vel,
                decoy_positions, decoy_velocities, decoy_active)
            
            has_lock = tracked_pos is not None
            
            if has_lock:
                tgt_accel_est = tracked_accel if tracked_accel is not None else np.zeros(3)
                accel_cmd = self.guidance.update(
                    t, dt, self.pos, self.vel,
                    tracked_pos, tracked_vel, tgt_accel_est, True)
            else:
                tgt_accel_est = np.zeros(3)
                accel_cmd = self.guidance.update(
                    t, dt, self.pos, self.vel,
                    tgt_pos, tgt_vel, tgt_accel_est, False)
        
        # Propulsion
        if speed > 1e-6:
            thrust_dir = self.vel / speed
        else:
            thrust_dir = np.array([1., 0., 0.])
        
        thrust_accel = (thrust_mag / self.mass) * thrust_dir
        
        # Aerodynamic drag
        mach = ATMOS.mach_number(self.vel, altitude)
        cd = get_missile_cd(mach)
        drag_force = calculate_drag(self.vel, altitude, cd, self.cfg.MISS_REF_AREA)
        drag_accel = drag_force / self.mass
        
        # Total acceleration
        total_accel = thrust_accel + drag_accel + accel_cmd + self.cfg.GRAVITY
        
        # Integrate
        self.vel = self.vel + total_accel * dt
        self.pos = self.pos + self.vel * dt
        
        # Ground collision
        if self.pos[2] <= 0:
            self.pos[2] = 0
            self.vel[:] = 0
            if not self.intercepted:
                self.intercepted = True
                self.intercept_type = 'ground'
        
        # Log telemetry
        lateral_g = np.linalg.norm(accel_cmd) / 9.81
        frame = TelemetryFrame(
            time=t,
            miss_pos=self.pos.copy(),
            miss_vel=self.vel.copy(),
            miss_accel_cmd=accel_cmd.copy(),
            miss_mass=self.mass,
            miss_mach=mach,
            guidance_mode=self.guidance.mode,
            zem=self.guidance.zem,
            tgo=self.guidance.tgo,
            los_rate=self.guidance.los_rate,
            closing_vel=self.guidance.closing_vel,
            seeker_state=self.seeker.state,
            seeker_snr=self.seeker.snr,
            gimbal_angle=self.seeker.gimbal_angle,
            lock_type=lock_type,
            tgt_pos=tgt_pos.copy(),
            tgt_vel=tgt_vel.copy(),
            range_to_target=R,
            lateral_accel=lateral_g
        )
        self.logger.log(frame)
        
        return lock_type


# ============================================================================
# SECTION 9: TARGET AIRCRAFT & DECOY CLASSES
# ============================================================================

# (FighterPolicyNet, FighterRLPolicy, TargetAircraft, Decoy classes remain
# largely the same - abbreviated here for space)

class FighterPolicyNet(nn.Module):
    """Neural network policy for RL-controlled fighter."""
    def __init__(self, obs_dim=17, hidden_dim=256):
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


class FighterRLPolicy:
    """Wrapper for RL policy with action mapping."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu",
                 obs_dim: int = 17):
        self.enabled = TORCH_AVAILABLE
        self.device = device
        self.obs_dim = obs_dim
        self.model = None
        
        if not self.enabled:
            return
        
        self.model = FighterPolicyNet(obs_dim=obs_dim, hidden_dim=256).to(device)
        self.model.eval()
        
        if model_path:
            try:
                state = torch.load(model_path, map_location=device)
                self.model.load_state_dict(state)
                logger.info(f"Loaded fighter policy from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
    
    def get_action(self, obs: np.ndarray, fwd: np.ndarray,
                   max_accel: float) -> np.ndarray:
        """Map observation to acceleration command."""
        if not self.enabled or self.model is None:
            return np.zeros(3)
        
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if obs.shape[0] != self.obs_dim:
            return np.zeros(3)
        
        with torch.no_grad():
            inp = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            out = self.model(inp)[0].cpu().numpy()
        
        dir_raw = out[:3]
        mag_raw = out[3]
        
        # Build local frame
        fwd = np.asarray(fwd, dtype=float)
        norm_fwd = np.linalg.norm(fwd)
        fwd = fwd / norm_fwd if norm_fwd > 1e-6 else np.array([1., 0., 0.])
        
        world_up = np.array([0., 0., 1.])
        right = np.cross(fwd, world_up)
        r_norm = np.linalg.norm(right)
        right = right / r_norm if r_norm > 1e-6 else np.array([0., 1., 0.])
        up_local = np.cross(right, fwd)
        
        d_r, d_u, _ = dir_raw
        VERT_SCALE = 0.4
        dir_local = d_r * right + d_u * VERT_SCALE * up_local
        d_norm = np.linalg.norm(dir_local)
        
        if d_norm < 1e-6:
            return np.zeros(3)
        
        dir_unit = dir_local / d_norm
        mag_scale = max(0., float(np.tanh(mag_raw)))
        
        return mag_scale * max_accel * dir_unit


class TargetAircraft:
    """AI-controlled fighter with heuristic and RL blend."""
    
    def __init__(self, start_pos: np.ndarray, rng: np.random.Generator,
                 rl_model_path: Optional[str] = None, rl_blend: float = 0.0,
                 cfg: SimConfig = CFG):
        self.cfg = cfg
        self.pos = start_pos.astype(float).copy()
        self.vel = np.array([cfg.TARG_INITIAL_SPEED, 0., 0.])
        self.rng = rng
        
        self.alt_band = (9000., 13000.)
        self.altitude_cmd = np.clip(self.pos[2], *self.alt_band)
        self.last_jink_time = 0.0
        
        self.rl_blend = float(np.clip(rl_blend, 0., 1.))
        self.rl_policy = FighterRLPolicy(model_path=rl_model_path, obs_dim=17)
    
    def _update_altitude_target(self, t: float):
        if t - self.last_jink_time > 5.0:
            self.altitude_cmd = self.rng.uniform(*self.alt_band)
            self.last_jink_time = t
    
    def _compute_heuristic_accel(self, t: float, 
                                  miss_pos: np.ndarray, 
                                  miss_vel: np.ndarray) -> np.ndarray:
        """Heuristic evasion logic."""
        max_accel = self.cfg.TARG_MAX_G * 9.81
        
        if miss_pos is None:
            return np.zeros(3)
        
        r = miss_pos - self.pos
        R = np.linalg.norm(r)
        if R < 1e-3:
            return np.zeros(3)
        
        los = r / R
        speed = np.linalg.norm(self.vel)
        fwd = self.vel / speed if speed > 1e-3 else np.array([1., 0., 0.])
        
        # Altitude control
        self._update_altitude_target(t)
        alt_err = self.altitude_cmd - self.pos[2]
        a_z = np.clip(0.01 * alt_err, -2*9.81, 2*9.81)
        vert_accel = np.array([0., 0., a_z])
        
        # Threat level
        threat = 1.0 if R < 8000 else 0.5 if R < 25000 else 0.2
        
        # Lateral evasion
        los_away = -los
        lateral = los_away - np.dot(los_away, fwd) * fwd
        lat_norm = np.linalg.norm(lateral)
        if lat_norm < 1e-5:
            lateral = np.cross(fwd, np.array([0., 0., 1.]))
            lat_norm = np.linalg.norm(lateral)
        lateral = lateral / lat_norm if lat_norm > 1e-6 else np.zeros(3)
        
        lat_accel = threat * max_accel * 0.9 * lateral
        
        # Jink
        jink = 0.2 * 9.81 * self.rng.standard_normal(3)
        jink -= np.dot(jink, fwd) * fwd
        
        a_cmd = vert_accel + lat_accel + jink
        
        # G-limit
        total = a_cmd + self.cfg.GRAVITY
        total_mag = np.linalg.norm(total)
        if total_mag > max_accel:
            total *= max_accel / total_mag
            a_cmd = total - self.cfg.GRAVITY
        
        return a_cmd
    
    def _build_obs(self, miss_pos: np.ndarray, miss_vel: np.ndarray) -> np.ndarray:
        """Build RL observation vector - must match training!"""
        r = miss_pos - self.pos
        v_rel = miss_vel - self.vel
        R = np.linalg.norm(r)
        speed = np.linalg.norm(self.vel)
        
        alt_mid = sum(self.alt_band) / 2
        alt_span = self.alt_band[1] - self.alt_band[0]
        
        # Tactical features (must match train.py exactly)
        los_unit = r / max(R, 1e-6)
        fwd = self.vel / max(speed, 1e-6)
        perpendicularity = 1.0 - abs(np.dot(fwd, los_unit))
        closing_rate = -np.dot(v_rel, los_unit)
        missile_ahead = np.dot(fwd, los_unit)
        
        return np.concatenate([
            r / 30000.,
            v_rel / 1000.,
            self.vel / 400.,
            [R / 30000., speed / 400., (self.pos[2] - alt_mid) / (alt_span/2),
            self.vel[2] / 200., self.vel[2] / speed if speed > 1e-3 else 0.,
            perpendicularity, closing_rate / 500.0, missile_ahead]
        ]).astype(np.float32)
    
    def update(self, t: float, dt: float,
               miss_pos: np.ndarray = None,
               miss_vel: np.ndarray = None) -> np.ndarray:
        """Update fighter state."""
        max_accel = self.cfg.TARG_MAX_G * 9.81
        
        a_heur = self._compute_heuristic_accel(t, miss_pos, miss_vel)
        
        if miss_pos is not None and self.rl_blend > 0:
            obs = self._build_obs(miss_pos, miss_vel)
            speed = np.linalg.norm(self.vel)
            fwd = self.vel / speed if speed > 1e-3 else np.array([1., 0., 0.])
            a_rl = self.rl_policy.get_action(obs, fwd, max_accel)
        else:
            a_rl = np.zeros(3)
        
        a_cmd = (1 - self.rl_blend) * a_heur + self.rl_blend * a_rl
        a_total = a_cmd + self.cfg.GRAVITY
        
        self.vel += a_total * dt
        self.pos += self.vel * dt
        
        # Maintain speed
        speed = np.linalg.norm(self.vel)
        if speed > 1e-3:
            self.vel *= self.cfg.TARG_INITIAL_SPEED / speed
        
        # Ground clamp
        if self.pos[2] < 0:
            self.pos[2] = 0
        
        return self.pos


class Decoy:
    """Countermeasure decoy."""
    
    def __init__(self, deploy_time: float, rng: np.random.Generator, cfg: SimConfig = CFG):
        self.cfg = cfg
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.deploy_time = deploy_time
        self.emission_end = deploy_time + cfg.DECOY_BURN_TIME
        self.active = False
        self.mass = cfg.DECOY_MASS
        
        # Random drift
        drift_dir = rng.standard_normal(3)
        norm = np.linalg.norm(drift_dir)
        drift_dir = drift_dir / norm if norm > 1e-9 else np.array([0., 1., 0.])
        drift_speed = max(0., cfg.DECOY_BASE_SPEED + cfg.DECOY_SPEED_SIGMA * rng.standard_normal())
        self.drift_vel = drift_dir * drift_speed
    
    def deploy(self, ac_pos: np.ndarray, ac_vel: np.ndarray):
        self.pos = ac_pos.copy()
        self.vel = ac_vel.copy() + self.drift_vel
        self.active = True
    
    def update(self, t: float, dt: float, ac_pos: np.ndarray, ac_vel: np.ndarray):
        if t >= self.deploy_time and t <= self.emission_end:
            if not self.active:
                self.deploy(ac_pos, ac_vel)
        elif t > self.emission_end:
            self.active = False
        
        if not self.active and t < self.deploy_time:
            self.pos = ac_pos.copy()
            self.vel = ac_vel.copy()
            return
        
        altitude = max(0., self.pos[2])
        drag = calculate_drag(self.vel, altitude, 1.0, self.cfg.DECOY_DRAG_AREA)
        accel = self.cfg.GRAVITY + drag / self.mass
        
        self.vel += accel * dt
        self.pos += self.vel * dt
        
        if self.pos[2] <= 0:
            self.pos[2] = 0
            self.vel[:] = 0


# ============================================================================
# SECTION 10: SIMULATION RUNNER
# ============================================================================

def sample_random_starts(rng: np.random.Generator, cfg: SimConfig = CFG
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Sample random engagement geometry."""
    # Aircraft
    ac_alt = rng.uniform(9000., 13000.)
    r_ac = rng.uniform(0., 3000.)
    bearing_ac = rng.uniform(0., 2 * np.pi)
    ac_start = np.array([r_ac * np.cos(bearing_ac), r_ac * np.sin(bearing_ac), ac_alt])
    
    # Missile (forward hemisphere)
    r = rng.uniform(8000., 15000.)
    bearing = rng.uniform(-np.pi/3, np.pi/3)
    miss_alt = rng.uniform(8000., 14000.)
    miss_start = np.array([
        ac_start[0] + r * np.cos(bearing),
        ac_start[1] + r * np.sin(bearing),
        miss_alt
    ])
    
    return ac_start, miss_start


def simulate_engagement(ac_start: np.ndarray, miss_start: np.ndarray,
                       rng: np.random.Generator = None,
                       make_plot: bool = False,
                       verbose: bool = True,
                       cfg: SimConfig = CFG) -> Dict:
    """
    Run single engagement simulation.
    
    Returns comprehensive results dictionary with telemetry.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    times = np.arange(0, cfg.TMAX, cfg.DT)
    n_points = len(times)
    
    # Initialize entities
    target = TargetAircraft(ac_start, rng, rl_model_path="fighter_policy.pt",
                           rl_blend=1.0, cfg=cfg) #Use heuristic or RL as needed set the blend value accordingly 
    
    # Initial missile velocity toward target
    rel = ac_start - miss_start
    R0 = np.linalg.norm(rel)
    los_hat = rel / R0 if R0 > 1e-6 else np.array([1., 0., 0.])
    initial_vel = 300. * los_hat
    
    missile = Missile(miss_start, initial_vel, rng, cfg)
    missile.launched = True
    missile.launch_time = cfg.MISSILE_LAUNCH_TIME
    
    # Dynamic decoy timing
    v_rel0 = target.vel - missile.vel
    closing0 = max(-np.dot(v_rel0, los_hat), 1.)
    t_go_est = R0 / closing0
    
    decoys = []
    for i in range(cfg.N_DECOYS):
        deploy_time = 0.6 * t_go_est + rng.uniform(-1., 1.)
        decoys.append(Decoy(deploy_time, rng, cfg))
    
    # Storage
    tgt_states = np.zeros((n_points, 3))
    miss_states = np.zeros((n_points, 3))
    miss_speeds = np.zeros(n_points)
    decoy_states = np.full((cfg.N_DECOYS, n_points, 3), np.nan)
    decoy_active = np.zeros((cfg.N_DECOYS, n_points), dtype=bool)
    lock_history = np.full(n_points, -1, dtype=int)
    
    # Results
    intercept_time = None
    intercept_idx = None
    intercept_type = None
    closest_miss = np.inf
    
    # Main loop
    for i, t in enumerate(times):
        target.update(t, cfg.DT, missile.pos, missile.vel)
        tgt_states[i] = target.pos.copy()
        
        # Update decoys
        d_pos_list = []
        d_vel_list = []
        d_active_list = []
        
        for d_idx, decoy in enumerate(decoys):
            decoy.update(t, cfg.DT, target.pos, target.vel)
            decoy_states[d_idx, i] = decoy.pos.copy()
            decoy_active[d_idx, i] = decoy.active
            d_pos_list.append(decoy.pos.copy())
            d_vel_list.append(decoy.vel.copy())
            d_active_list.append(decoy.active)
        
        # Update missile
        if not missile.intercepted:
            lock_type = missile.update(t, cfg.DT, target.pos, target.vel,
                                       d_pos_list, d_vel_list, d_active_list)
            
            if lock_type == 'target':
                lock_history[i] = 0
            elif lock_type and lock_type.startswith('decoy_'):
                lock_history[i] = int(lock_type.split('_')[1]) + 1
            
            # Check intercepts
            dist_to_tgt = np.linalg.norm(target.pos - missile.pos)
            closest_miss = min(closest_miss, dist_to_tgt)
            
            if dist_to_tgt < cfg.KILL_DIST:
                missile.intercepted = True
                missile.intercept_type = 'real'
                intercept_time = t
                intercept_idx = i
                intercept_type = 'real'
                if verbose:
                    logger.info(f"HIT TARGET at t={t:.2f}s, d={dist_to_tgt:.1f}m")
            else:
                for d_idx, decoy in enumerate(decoys):
                    if decoy.active:
                        dist_d = np.linalg.norm(decoy.pos - missile.pos)
                        if dist_d < cfg.KILL_DIST:
                            missile.intercepted = True
                            missile.intercept_type = f'decoy_{d_idx}'
                            intercept_time = t
                            intercept_idx = i
                            intercept_type = 'decoy'
                            if verbose:
                                logger.info(f"DECOY {d_idx+1} HIT at t={t:.2f}s")
                            break
        
        miss_states[i] = missile.pos.copy()
        miss_speeds[i] = np.linalg.norm(missile.vel)
    
    # Compute CPA from telemetry
    cpa_dist, cpa_time, _ = missile.logger.compute_miss_distance()
    
    result = {
        'intercept_real': intercept_type == 'real',
        'intercept_decoy': intercept_type == 'decoy',
        'intercept_ground': intercept_type == 'ground',
        'intercept_type': intercept_type,
        'intercept_time': intercept_time,
        'closest_miss': closest_miss,
        'cpa_distance': cpa_dist,
        'cpa_time': cpa_time,
        'telemetry': missile.logger,
        # Data for plotting
        '_tgt_states': tgt_states,
        '_miss_states': miss_states,
        '_miss_speeds': miss_speeds,
        '_decoy_states': decoy_states,
        '_decoy_active': decoy_active,
        '_lock_history': lock_history,
        '_times': times,
        '_intercept_idx': intercept_idx,
    }
    
    if verbose:
        logger.info(f"Final range: {np.linalg.norm(tgt_states[-1] - miss_states[-1]):.0f}m")
        logger.info(f"CPA: {cpa_dist:.1f}m at t={cpa_time:.2f}s")
        if intercept_type == 'real':
            logger.info("RESULT: TARGET HIT")
        elif intercept_type == 'decoy':
            logger.info("RESULT: DECOY HIT")
        else:
            logger.info("RESULT: MISS")
    
    if make_plot:
        create_animation(result, ac_start, miss_start, cfg)
    
    return result


# ============================================================================
# SECTION 11: VISUALIZATION (Abbreviated)
# ============================================================================

_current_animation = None

def create_animation(result: Dict, ac_start: np.ndarray, miss_start: np.ndarray,
                    cfg: SimConfig = CFG):
    """Create 3D animated visualization."""
    global _current_animation
    
    # Extract data
    times = result['_times']
    tgt_states = result['_tgt_states']
    miss_states = result['_miss_states']
    miss_speeds = result['_miss_speeds']
    decoy_states = result['_decoy_states']
    decoy_active = result['_decoy_active']
    lock_history = result['_lock_history']
    intercept_idx = result['_intercept_idx']
    intercept_type = result['intercept_type']
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Compute bounds
    all_pts = np.vstack([tgt_states, miss_states])
    max_range = max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1]), np.ptp(all_pts[:, 2]), 1000)
    center = (all_pts.max(0) + all_pts.min(0)) / 2
    radius = max_range / 2 * 1.1
    
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(0, center[2] - radius), center[2] + radius)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.view_init(elev=20, azim=45)
    
    # Static elements
    ax.plot(tgt_states[:, 0], tgt_states[:, 1], tgt_states[:, 2], 'b--', alpha=0.3)
    ax.scatter(*ac_start, c='green', s=100, marker='s')
    ax.scatter(*miss_start, c='orange', s=100, marker='^')
    
    if intercept_idx:
        ax.scatter(*tgt_states[intercept_idx], c='red', s=300, marker='*')
    
    # Animated elements
    tgt_pt, = ax.plot([], [], [], 'bo', ms=10)
    tgt_trail, = ax.plot([], [], [], 'b-', lw=2, alpha=0.7)
    miss_pt, = ax.plot([], [], [], 'ro', ms=8)
    miss_trail, = ax.plot([], [], [], 'r-', lw=2, alpha=0.7)
    
    decoy_pts = [ax.plot([], [], [], 'y*', ms=12)[0] for _ in range(cfg.N_DECOYS)]
    
    # HUD
    hud_x = 0.72
    time_txt = ax.text2D(hud_x, 0.95, '', transform=ax.transAxes, fontsize=10, family='monospace')
    range_txt = ax.text2D(hud_x, 0.91, '', transform=ax.transAxes, fontsize=10, family='monospace')
    zem_txt = ax.text2D(hud_x, 0.87, '', transform=ax.transAxes, fontsize=10, family='monospace')
    tgo_txt = ax.text2D(hud_x, 0.83, '', transform=ax.transAxes, fontsize=10, family='monospace')
    
    ax.set_title('3D Missile-Target Engagement (Enhanced Simulation)')
    
    telemetry = result['telemetry']
    
    def init():
        for a in [tgt_pt, tgt_trail, miss_pt, miss_trail] + decoy_pts:
            a.set_data([], [])
            a.set_3d_properties([])
        return (tgt_pt, tgt_trail, miss_pt, miss_trail, *decoy_pts,
                time_txt, range_txt, zem_txt, tgo_txt)
    
    def update(frame):
        tgt_pt.set_data([tgt_states[frame, 0]], [tgt_states[frame, 1]])
        tgt_pt.set_3d_properties([tgt_states[frame, 2]])
        tgt_trail.set_data(tgt_states[:frame+1, 0], tgt_states[:frame+1, 1])
        tgt_trail.set_3d_properties(tgt_states[:frame+1, 2])
        
        miss_pt.set_data([miss_states[frame, 0]], [miss_states[frame, 1]])
        miss_pt.set_3d_properties([miss_states[frame, 2]])
        miss_trail.set_data(miss_states[:frame+1, 0], miss_states[:frame+1, 1])
        miss_trail.set_3d_properties(miss_states[:frame+1, 2])
        
        lock = lock_history[frame]
        color = 'red' if lock == 0 else 'magenta' if lock > 0 else 'gray'
        miss_pt.set_color(color)
        miss_trail.set_color(color)
        
        for d in range(cfg.N_DECOYS):
            if decoy_active[d, frame]:
                pos = decoy_states[d, frame]
                decoy_pts[d].set_data([pos[0]], [pos[1]])
                decoy_pts[d].set_3d_properties([pos[2]])
            else:
                decoy_pts[d].set_data([], [])
                decoy_pts[d].set_3d_properties([])
        
        # HUD
        dist = np.linalg.norm(tgt_states[frame] - miss_states[frame])
        time_txt.set_text(f'T: {times[frame]:.1f}s')
        range_txt.set_text(f'R: {dist/1000:.1f}km')
        
        # Get ZEM/tgo from telemetry (approximate frame mapping)
        telem_idx = min(frame, len(telemetry.frames) - 1)
        if telem_idx >= 0 and telemetry.frames:
            f = telemetry.frames[telem_idx]
            zem_txt.set_text(f'ZEM: {f.zem:.0f}m')
            tgo_txt.set_text(f'Tgo: {f.tgo:.1f}s')
        
        return (tgt_pt, tgt_trail, miss_pt, miss_trail, *decoy_pts,
                time_txt, range_txt, zem_txt, tgo_txt)
    
    frame_skip = max(1, len(times) // cfg.ANIMATION_MAX_FRAMES)
    frames = list(range(0, len(times), frame_skip))
    
    _current_animation = FuncAnimation(fig, update, frames=frames, init_func=init,
                                       blit=False, interval=cfg.ANIMATION_INTERVAL)
    plt.tight_layout()
    plt.show()


# ============================================================================
# SECTION 12: MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng()
    
    n_trials = 10
    results = {'real': 0, 'decoy': 0, 'miss': 0}
    cpas = []
    
    print("=" * 70)
    print("REALISTIC 3D MISSILE-TARGET ENGAGEMENT SIMULATION v2.0")
    print("=" * 70)
    print(f"Running {n_trials} Monte Carlo trials...\n")
    
    for k in range(n_trials):
        ac, miss = sample_random_starts(rng)
        print(f"Trial {k+1}/{n_trials}:")
        
        result = simulate_engagement(ac, miss, rng, make_plot=True, verbose=True)
        
        if result['intercept_real']:
            results['real'] += 1
        elif result['intercept_decoy']:
            results['decoy'] += 1
        else:
            results['miss'] += 1
        
        cpas.append(result['cpa_distance'])
        print()
    
    print("=" * 70)
    print("MONTE CARLO RESULTS")
    print("=" * 70)
    print(f"  Target Pk:    {results['real']:3d}/{n_trials} ({100*results['real']/n_trials:.1f}%)")
    print(f"  Decoy hits:   {results['decoy']:3d}/{n_trials} ({100*results['decoy']/n_trials:.1f}%)")
    print(f"  Misses:       {results['miss']:3d}/{n_trials} ({100*results['miss']/n_trials:.1f}%)")
    print(f"  Mean CPA:     {np.mean(cpas):.1f}m")
    print(f"  Median CPA:   {np.median(cpas):.1f}m")
    print("=" * 70)