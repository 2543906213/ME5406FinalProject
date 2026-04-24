"""
env.py — Reinforcement Learning Environment for Franka Panda Hole-Piercing Task
==============================================================================
Task Description:
    A Franka Panda robotic arm (7-DOF: all joints controlled, gripper ignored) 
    starts from an initial position, passes through a vertically placed board 
    with a hole, and reaches a target point behind the board.
    The robot must align with the hole to pass through; any collision with 
    the board terminates the episode.

External Interface (used by train.py / ppo.py):
----------------------------------------------
    env = HoleBoardEnv(cfg: EnvConfig)

    scalar_obs, wrist_depth, global_depth, info = env.reset()
        scalar_obs   : np.ndarray  shape=(SCALAR_DIM,)  float32   Scalar observations
        wrist_depth  : np.ndarray  shape=(IMG_H, IMG_W)  float32   Normalized to [0,1]
        global_depth : np.ndarray  shape=(IMG_H, IMG_W)  float32   Normalized to [0,1]
        info         : dict        {'hole_center': (y,z), 'target_pos': (x,y,z)}

    scalar_obs, wrist_depth, global_depth, reward, done, info = env.step(action)
        action       : np.ndarray  shape=(ACTION_DIM,)  float32   Scaled in [-1,1]
        reward       : float
        done         : bool
        info         : dict        {'success': bool, 'collision': bool, 'timeout': bool,
                                    'ee_pos': np.ndarray, 'dist_to_target': float}

    env.close()

Constants for network.py:
    SCALAR_DIM     = 31    # Actor scalar obs dim: joints(18)+EE(7)+target(4)+passed(1)+alive(1)
    PRIVILEGED_DIM = 48    # Critic privileged obs (pure scalar, no CNN)
    IMG_H, IMG_W   = 64, 64 # Depth map resolution (pixels)
    ACTION_DIM     = 6     # Action dimension (6-DOF, excluding joint7 rotation)
    WRIST_DEPTH_MAX  = 2.0 # Wrist camera depth clip limit (meters)
    GLOBAL_DEPTH_MAX = 5.0 # Global camera depth clip limit (meters)
"""

# -- Simulation must be started at the first line -----------------------------
from common.normalization import Normalization
from isaacsim import SimulationApp
# When headless=True, the UI is not displayed, which speeds up training.
# Controlled by EnvConfig.headless; placeholder used here, passed during instantiation.
# Note: SimulationApp can only be created once globally.
import sys
import os
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# -- Dimension constants for network.py ---------------------------------------
# SCALAR_DIM = 31
SCALAR_DIM = 27
PRIVILEGED_DIM = 48
IMG_H = 64
IMG_W = 64
# 6-DOF control (excluding joint7 wrist rotation and gripper)
ACTION_DIM = 5
# ACTION_DIM = 6
WRIST_DEPTH_MAX = 2.0
GLOBAL_DEPTH_MAX = 5.0

# -- Joint Indices ------------------------------------------------------------
# Franka arm: first 6 joints (joint7 and fingers fixed)
ACTIVE_JOINT_IDX = [0, 1, 2, 3, 4, 5]
N_JOINTS_TOTAL = 9                  # Total Franka joints (7 arm + 2 gripper)


# -----------------------------------------------------------------------------
# Configuration Class
# -----------------------------------------------------------------------------
@dataclass
class EnvConfig:
    # -- Simulation -----------------------------------------
    headless: bool = False          # True = No UI, faster training
    physics_dt: float = 1.0 / 60.0   # Physics time step (sec)
    rendering_dt: float = 1.0 / 60.0

    # -- Scene Geometry (Meters) ----------------------------
    board_x: float = 0.1            # Board position on X-axis
    board_thickness: float = 0.02    # Board thickness
    board_half_y: float = 0.8         # Board half-width (Y direction)
    board_z_low: float = 0.20       # Board bottom Z coordinate
    board_z_high: float = 1.20       # Board top Z coordinate

    # Hole randomization range (offset within the board plane)
    hole_y_range: Tuple[float, float] = (-0.1, 0.1)   # Hole center Y range
    hole_z_range: Tuple[float, float] = (0.50, 0.60)  # Hole center Z range
    # Square hole half-side length (~30cm total)
    hole_half_size: float = 0.15

    # Target point (behind the board)
    target_x_range: Tuple[float, float] = (0.30, 0.50)  # Target X range
    # Target Y relative to hole
    target_y_range: Tuple[float, float] = (-0.05, 0.05)
    # Target Z relative to hole
    target_z_range: Tuple[float, float] = (-0.05, 0.05)

    # -- Initial Joint Angles (Radians) ---------------------
    init_joints: list = field(default_factory=lambda: [
        0.0,    # panda_joint1
        -1.7,    # panda_joint2
        0.0,    # panda_joint3
        -2.356,  # panda_joint4
        0.0,    # panda_joint5
        1.571,  # panda_joint6
        0.785,  # panda_joint7
        0.01,   # panda_finger_joint1
        0.01,   # panda_finger_joint2
    ])

    # -- Action ---------------------------------------------
    # Map [-1,1] actions to joint increments (rad/step)
    action_scale: float = 0.20

    # -- Curriculum Stage (Affects reward weights, set via set_stage()) ----
    stage: int = 1   # 1 / 2 / 3 / 4

    # -- Termination Conditions -----------------------------
    max_steps: int = 500       # Maximum steps per episode (timeout)
    success_dist: float = 0.06   # Success if distance < this (after piercing)
    workspace_limit: float = 1.5  # Out-of-bounds if distance from origin > this

    # -- Cameras --------------------------------------------
    # Wrist Camera: Mounted on ee_link, moves with EE
    wrist_cam_offset: Tuple = (0.0, 0.0, 0.1)

    # Global Camera: Fixed in scene, overlooking workspace
    global_cam_pos:  Tuple = (1.0, 0.0, 1.2)
    global_cam_look: Tuple = (0.40, 0.0, 0.55)

    # -- Debug ----------------------------------------------
    debug_diagnostics: bool = False
    debug_log_episodes: int = 3
    debug_log_steps: int = 5


# -----------------------------------------------------------------------------
# Termination Manager
# -----------------------------------------------------------------------------
class TerminationManager:
    """
    Manages all episode termination conditions. Called in step().

    Conditions:
      Success    — EE passed the board AND distance to target < success_dist
      Collision  — Robotic arm touches the board
      Timeout    — step_count >= max_steps
      Out-of-WS  — ||ee_pos|| > workspace_limit
    """

    def __init__(self, cfg: EnvConfig):
        self._success_dist = cfg.success_dist
        self._max_steps = cfg.max_steps
        self._workspace_limit = cfg.workspace_limit

    def check(
        self,
        step_count:   int,
        ee_pos:       np.ndarray,
        passed_board: bool,
        target_pos:   np.ndarray,
        collided:     bool,
    ) -> Tuple[bool, dict]:
        dist_to_target = float(np.linalg.norm(ee_pos - target_pos))

        success = passed_board and (dist_to_target < self._success_dist)
        timeout = step_count >= self._max_steps
        out_of_ws = float(np.linalg.norm(ee_pos)) > self._workspace_limit

        done = success or collided or timeout or out_of_ws

        return done, {
            "success":          success,
            "collision":        collided,
            "timeout":          timeout,
            "out_of_workspace": out_of_ws,
            "dist_to_target":   dist_to_target,
            "step_count":       step_count,
        }


# -----------------------------------------------------------------------------
# Reward Manager
# -----------------------------------------------------------------------------
class RewardManager:
    """
    Reward function manager: Terms are implemented as independent methods 
    and summed with weights based on curriculum stages.

    Terms:
      R_approach   — Potential difference before piercing, φ = -d(EE, hole_center)
      R_target     — Potential difference after piercing, φ = -d(EE, target)
      R_tguide     — YZ plane alignment guide toward target before/after piercing
      R_align      — EE velocity alignment with +X axis (active near board)
      R_pass       — Sparse reward for piercing (clean vs grazing)
      R_collision  — Penalty for soft contact / hard collision
      R_smooth     — Action smoothness penalty -||Δa||²
      R_arrive     — One-time reward for reaching target
    """

    _STAGE_CFG = {
        1: dict(w_progress=10.0, w_target_guide=5.0,  w_align=0.005, w_post_align=0.0,
                r_pass_clean=5.0, r_pass_grazing=4.0,
                r_soft_coll=-0.1,  r_hard_coll=-0.5,
                w_smooth=0.0,      r_arrive=10.0,  r_alive=-0.005),

        2: dict(w_progress=8.0,  w_target_guide=10.0,  w_align=0.005, w_post_align=0.005,
                r_pass_clean=5.0, r_pass_grazing=2.0,
                r_soft_coll=-0.2,  r_hard_coll=-0.8,
                w_smooth=0.0,      r_arrive=10.0,  r_alive=-0.005),

        3: dict(w_progress=8.0,  w_target_guide=12.0, w_align=0.012,  w_post_align=0.008,
                r_pass_clean=5.0, r_pass_grazing=1.0,
                r_soft_coll=-0.5,  r_hard_coll=-1.0,
                w_smooth=0.001,    r_arrive=10.0,  r_alive=-0.007),

        4: dict(w_progress=5.0,  w_target_guide=15.0, w_align=0.015,  w_post_align=0.01,
                r_pass_clean=5.0, r_pass_grazing=0.1,
                r_soft_coll=-0.5,  r_hard_coll=-2.0,
                w_smooth=0.005,    r_arrive=10.0,  r_alive=-0.01),
    }

    def __init__(self, cfg: 'EnvConfig', stage: int = 1):
        self.cfg = cfg
        self.stage = stage
        self._prev_phi_approach = 0.0   # Baseline for φ_approach
        self._prev_phi_target = 0.0   # Baseline for φ_target
        self._prev_phi_tguide = 0.0   # Baseline for φ_tguide
        self._arrived = False
        self._pass_rewarded = False
        self._hole_center = np.zeros(3, dtype=np.float32)
        self.last_breakdown = {"progress": 0., "align": 0., "post_align": 0., "pass": 0.,
                               "target_guide": 0., "collision": 0., "smooth": 0., "arrive": 0., "alive": 0.}

    def set_stage(self, stage: int):
        assert stage in (1, 2, 3, 4), f"Stage must be 1/2/3/4, got {stage}"
        self.stage = stage

    def reset(self, ee_pos: np.ndarray, hole_center: np.ndarray,
              target_pos: np.ndarray):
        """Reset potential baselines and flags at the start of each episode."""
        self._prev_phi_approach = -float(np.linalg.norm(ee_pos - hole_center))
        self._prev_phi_target = -float(np.linalg.norm(ee_pos - target_pos))
        self._prev_phi_tguide = - \
            float(np.linalg.norm((ee_pos - target_pos)[1:3]))
        self._arrived = False
        self._pass_rewarded = False

    def _r_approach(self, ee_pos: np.ndarray, hole_center: np.ndarray,
                    passed_board: bool) -> float:
        """Potential diff before piercing: φ = -d(EE, hole). Frozen after piercing."""
        if passed_board:
            return 0.0
        phi = -float(np.linalg.norm(ee_pos - hole_center))
        r = phi - self._prev_phi_approach
        self._prev_phi_approach = phi
        return r

    def _r_target(self, ee_pos: np.ndarray, target_pos: np.ndarray,
                  passed_board: bool, just_passed: bool) -> float:
        """Potential diff after piercing: φ = -d(EE, target). Baseline reset at piercing moment."""
        if not passed_board:
            return 0.0
        phi = -float(np.linalg.norm(ee_pos - target_pos))
        if just_passed:   # Avoid reward spike at the frame of piercing
            self._prev_phi_target = phi
            return 0.0
        r = phi - self._prev_phi_target
        self._prev_phi_target = phi
        return r

    def _r_align(self, ee_pos: np.ndarray, ee_vel: np.ndarray,
                 passed_board: bool = False) -> float:
        """cos(θ) − 1, where θ is angle between EE velocity and +X axis. Active near board."""
        if passed_board:
            return 0.0
        if abs(ee_pos[0] - self.cfg.board_x) >= 0.05:
            return 0.0
        speed = float(np.linalg.norm(ee_vel)) + 1e-8
        cos_theta = float(np.clip(ee_vel[0] / speed, -1.0, 1.0))
        return cos_theta - 1.0

    def _r_post_align(self, ee_vel: np.ndarray, ee_pos: np.ndarray,
                      target_pos: np.ndarray, passed_board: bool) -> float:
        """After piercing: encourage velocity alignment with vector to target."""
        if not passed_board:
            return 0.0
        to_target = target_pos - ee_pos
        dist = float(np.linalg.norm(to_target)) + 1e-8
        speed = float(np.linalg.norm(ee_vel)) + 1e-8
        cos_theta = float(
            np.clip(np.dot(ee_vel, to_target) / (speed * dist), -1.0, 1.0))
        return cos_theta - 1.0

    def _r_target_guide(self, ee_pos: np.ndarray, target_pos: np.ndarray,
                        passed_board: bool, just_passed: bool) -> float:
        """Guide reward: Potential diff in YZ plane to encourage lateral alignment."""
        if not passed_board:
            return 0.0
        delta_yz = (ee_pos - target_pos)[1:3]
        phi = -(abs(float(delta_yz[0])) + abs(float(delta_yz[1])))
        if just_passed:
            self._prev_phi_tguide = phi
            return 0.0
        r = phi - self._prev_phi_tguide
        self._prev_phi_tguide = phi
        return r

    def _r_pass(self, just_passed: bool, ee_pos: np.ndarray) -> float:
        """Sparse piercing reward: Clean (+15) vs Grazing (+5)."""
        if self._pass_rewarded or not just_passed:
            return 0.0
        self._pass_rewarded = True
        cfg = self.cfg
        h = cfg.hole_half_size
        hy, hz = self._hole_center[1], self._hole_center[2]
        dy, dz = abs(ee_pos[1] - hy), abs(ee_pos[2] - hz)
        dist_to_edge = min(h - dy, h - dz)
        w = self._STAGE_CFG[self.stage]
        return w['r_pass_grazing'] if dist_to_edge < 0.05 else w['r_pass_clean']

    def _r_collision(self, ee_pos: np.ndarray, collided: bool) -> float:
        """Collision penalty: Hard collision vs Soft contact (near hole edges)."""
        w = self._STAGE_CFG[self.stage]
        if collided:
            return w['r_hard_coll']

        cfg = self.cfg
        half_th = cfg.board_thickness / 2.0
        if abs(ee_pos[0] - cfg.board_x) > half_th + 0.02:
            return 0.0
        if not (-cfg.board_half_y < ee_pos[1] < cfg.board_half_y):
            return 0.0
        if not (cfg.board_z_low < ee_pos[2] < cfg.board_z_high):
            return 0.0

        h = cfg.hole_half_size
        hy, hz = self._hole_center[1], self._hole_center[2]
        dy, dz = abs(ee_pos[1] - hy), abs(ee_pos[2] - hz)
        in_hole = (dy < h) and (dz < h)
        if not in_hole:
            return 0.0
        dist_to_edge = min(h - dy, h - dz)
        return w['r_soft_coll'] if dist_to_edge < 0.02 else 0.0

    def _r_smooth(self, action: np.ndarray, prev_action: np.ndarray) -> float:
        """Smoothness penalty: -||a_t − a_{t-1}||²"""
        return -float(np.sum((action - prev_action) ** 2))

    def _r_arrive(self, ee_pos: np.ndarray, target_pos: np.ndarray,
                  passed_board: bool) -> float:
        """One-time reward for reaching target distance after piercing."""
        if self._arrived:
            return 0.0
        if passed_board and float(np.linalg.norm(ee_pos - target_pos)) < self.cfg.success_dist:
            self._arrived = True
            return 1.0
        return 0.0

    def compute(self, ee_pos, ee_vel, action, prev_action, hole_center, target_pos, passed_board, collided, just_passed) -> float:
        """Main reward computation: returns the weighted sum of all terms."""
        w = self._STAGE_CFG[self.stage]
        self._hole_center = hole_center

        r_progress = w['w_progress'] * (
            self._r_approach(ee_pos, hole_center, passed_board) +
            self._r_target(ee_pos, target_pos, passed_board, just_passed)
        )
        r_tguide = w['w_target_guide'] * \
            self._r_target_guide(ee_pos, target_pos, passed_board, just_passed)
        r_align = w['w_align'] * self._r_align(ee_pos, ee_vel, passed_board)
        r_post_align = w['w_post_align'] * \
            self._r_post_align(ee_vel, ee_pos, target_pos, passed_board)
        r_pass = self._r_pass(just_passed, ee_pos)
        r_coll = self._r_collision(ee_pos, collided)
        r_smooth = w['w_smooth'] * self._r_smooth(action, prev_action)
        r_arrive = w['r_arrive'] * \
            self._r_arrive(ee_pos, target_pos, passed_board)
        r_alive = w['r_alive']    # Constant survival penalty

        self.last_breakdown = {
            "progress":    r_progress,
            "target_guide": r_tguide,
            "align":       r_align,
            "post_align":  r_post_align,
            "pass":        r_pass,
            "collision":   r_coll,
            "smooth":      r_smooth,
            "arrive":      r_arrive,
            "alive":       r_alive,
        }
        return float(sum(self.last_breakdown.values()))


# -----------------------------------------------------------------------------
# Main Environment Class
# -----------------------------------------------------------------------------
class HoleBoardEnv:
    """
    Reinforcement Learning environment for UR10e hole-piercing task (Isaac Sim standalone mode).

    Usage (in train.py):
        cfg = EnvConfig(headless=True)
        env = HoleBoardEnv(cfg)
        obs, depth_wrist, depth_global, info = env.reset()
        obs, depth_wrist, depth_global, reward, done, info = env.step(action)
        env.close()
    """

    def __init__(self, cfg: EnvConfig, sim_app: Optional[SimulationApp] = None):
        """
        Parameters
        ----------
        cfg     : EnvConfig      Hyperparameter configuration
        sim_app : SimulationApp  If already created in train.py, pass it directly to avoid 
                                 double initialization. If None, it will be created here 
                                 (suitable for debugging env.py independently).
        """
        self.cfg = cfg

        # -- Start Simulation App -----------------------------------------------
        # Only one instance of SimulationApp is allowed globally.
        if sim_app is None:
            self._sim_app = SimulationApp({
                "headless": cfg.headless,
                "renderer": "RayTracedLighting",
            })
        else:
            self._sim_app = sim_app  # Use external instance

        # -- Late Imports (Must import after SimulationApp starts) --------------
        self._late_imports()

        # -- Create Isaac Sim World ---------------------------------------------
        from omni.isaac.core import World
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()

        # -- Runtime States -----------------------------------------------------
        self._step_count = 0             # Current episode step count
        self._prev_action = np.zeros(
            # Previous action (for smoothness penalty)
            ACTION_DIM, dtype=np.float32)
        # Previous EE X-coordinate (for piercing detection)
        self._prev_ee_x = 0.0
        self._passed_board = False       # Flag indicating if the board has been pierced
        self._episode_count = 0
        # Curriculum stage change request (applies on next reset)
        self._pending_stage = None

        # Curriculum Statistics (Rolling window of 30 episodes)
        # Tracks success rates of piercing and reaching target for evaluate_policy monitoring.
        self._curriculum_window = 30
        # Record of _passed_board
        self._pass_history = deque(maxlen=self._curriculum_window)
        # Record of term_info["success"]
        self._arrive_history = deque(maxlen=self._curriculum_window)

        # Current episode's hole center and target coordinates (randomized per reset)
        self._hole_center = np.zeros(3, dtype=np.float32)   # (x=board_x, y, z)
        self._target_pos = np.zeros(3, dtype=np.float32)    # (x, y, z)

        # Board prim references (populated in _build_scene, updated in _reset_board)
        self._board_pieces = []   # list of FixedCuboid

        # -- Scene Construction -------------------------------------------------
        self._build_scene()

        # Must call reset once after all scene.add() calls to initialize physics engine
        self.world.reset()
        self._sim_app.update()   # Drive one frame to initialize OmniGraph pipelines
        self._setup_camera()     # annotator.attach() requires rendering pipeline readiness

        # -- PhysX Tensor API (Lazy Initialization) -----------------------------
        # Must be initialized after at least one world.step() call.
        # Indices are determined from actual PhysX link names to ensure USD/PhysX consistency.
        self._physics_sim_view = None
        self._art_physx_view = None
        self._ee_link_physx_idx = None     # Populated by _init_physx_art_view()
        self._arm_link_physx_idxs = None   # Populated by _init_physx_art_view()

        # EE Velocity Estimation (for privileged observation cos(theta))
        self._prev_ee_pos = np.zeros(3, dtype=np.float32)

        # -- Observation Normalizers --------------------------------------------
        self.scalar_norm = Normalization(shape=SCALAR_DIM)
        self.priv_norm = Normalization(shape=PRIVILEGED_DIM)
        self.training = True   # When False, normalizers do not update statistics

        # -- Managers -----------------------------------------------------------
        self._term_manager = TerminationManager(cfg)
        self._reward_manager = RewardManager(cfg, stage=cfg.stage)

    # --------------------------------------------------------------------------
    # Late Imports
    # --------------------------------------------------------------------------
    def _late_imports(self):
        """Centralize Isaac Sim / omni related imports to prevent errors before SimulationApp start."""
        global np
        import omni.replicator.core as rep
        import omni.usd
        from omni.isaac.core.objects import FixedCuboid, VisualSphere, VisualCuboid
        from omni.isaac.core.robots import Robot
        from omni.isaac.core.prims import XFormPrimView
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.core.utils.types import ArticulationAction
        from isaacsim.storage.native import get_assets_root_path
        from pxr import UsdGeom, Gf, Usd
        import omni.physics.tensors.impl.api as physx
        from isaacsim.core.simulation_manager import SimulationManager

        # Store to self for use in subsequent methods
        self._rep = rep
        self._omni_usd = omni.usd
        self._UsdGeom = UsdGeom
        self._Gf = Gf
        self._Usd = Usd
        self._physx = physx
        self._SimulationManager = SimulationManager
        self._FixedCuboid = FixedCuboid
        self._VisualSphere = VisualSphere
        self._XFormPrimView = XFormPrimView
        self._Robot = Robot
        self._add_ref = add_reference_to_stage
        self._ArtAction = ArticulationAction
        self._assets_root = get_assets_root_path()

    # --------------------------------------------------------------------------
    # Scene Setup (Called once during __init__)
    # --------------------------------------------------------------------------
    def _build_scene(self):
        """
        Builds the static scene structure:
          1. Franka Panda robotic arm
          2. Board with hole (4 FixedCuboids forming a frame, randomized during reset)
          3. Target marker (visual sphere)
        Note: Board position/size is updated every episode in _reset_board().
        """
        cfg = self.cfg

        # -- 1. Load Franka Panda ------------------------------------------------
        franka_usd = self._assets_root + \
            "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
        self._add_ref(usd_path=franka_usd, prim_path="/World/Franka")

        # Robot wrapper: provides interfaces like get_joint_positions / apply_action
        self._robot = self._Robot(prim_path="/World/Franka", name="franka")
        self.world.scene.add(self._robot)

        # End-Effector (EE) view: Standard Isaac Sim prims API
        self._ee_prim_path = "/World/Franka/panda_hand"
        self._ee_view = None   # Initialized after world.reset()

        # -- Disable Gripper Collisions ------------------------------------------
        # Fingers are not needed for the piercing task; disable to prevent false collisions.
        from omni.isaac.core.utils.stage import get_current_stage
        from pxr import UsdPhysics, UsdGeom
        _stage = get_current_stage()
        for _finger_path in ["/World/Franka/panda_hand", "/World/Franka/panda_leftfinger", "/World/Franka/panda_rightfinger"]:
            _prim = _stage.GetPrimAtPath(_finger_path)
            if _prim.IsValid():
                UsdPhysics.CollisionAPI.Apply(
                    _prim).GetCollisionEnabledAttr().Set(False)
                UsdGeom.Imageable(_prim).MakeInvisible()

        # -- 2. Place Board with Hole (4-Piece Framework) ------------------------
        # Board is at x = board_x, expanding on the YZ plane.
        # Framework consists of: Top, Bottom, Left, and Right pieces.
        # Placeholders are used here; scales and positions are updated during reset.
        placeholder_pos = np.array([cfg.board_x, 0.0, 0.55])
        placeholder_size = np.array([cfg.board_thickness, 0.01, 0.01])

        for name in ["Top", "Bottom", "Left", "Right"]:
            piece = self._FixedCuboid(
                prim_path=f"/World/Board/{name}",
                name=f"board_{name.lower()}",
                position=placeholder_pos.copy(),
                scale=placeholder_size.copy(),
                color=np.array([0.6, 0.6, 0.6]),
            )
            self.world.scene.add(piece)
            self._board_pieces.append(piece)

        # -- 3. Target Visualization (No collision, visual only) -----------------
        self._target_marker = self._VisualSphere(
            prim_path="/World/Target",
            radius=0.03,
            color=np.array([1.0, 0.5, 0.0]),
        )
        self.world.scene.add(self._target_marker)

    # --------------------------------------------------------------------------
    # Depth Camera Setup (Called once during __init__)
    # --------------------------------------------------------------------------
    def _setup_camera(self):
        """
        Dual Camera System:
          1. Wrist Camera  —— Mounted on ee_link, moves with EE, clip [0.01, 2.0m]
          2. Global Camera —— Fixed, overlooks the workspace, clip [0.01, 5.0m]

        Depth Annotator (distance_to_image_plane):
          - Orthogonal distance per pixel to image plane (meters), shape (H, W) float32
          - Invalid/Inf pixels are handled in _normalize_depth()
        """
        rep = self._rep
        cfg = self.cfg

        # Disable automatic capture; triggered manually in step()/reset()
        rep.orchestrator.set_capture_on_play(False)

        # -- 1. Global Camera (Fixed scene frame) --------------------------------
        self._global_camera = rep.create.camera(
            position=cfg.global_cam_pos,
            look_at=cfg.global_cam_look,
            focal_length=18.0,
            horizontal_aperture=24.0,
            clipping_range=(0.01, GLOBAL_DEPTH_MAX),
            name="GlobalCamera",
        )
        global_rp = rep.create.render_product(
            self._global_camera, (IMG_W, IMG_H), name="GlobalView", force_new=True
        )
        self._global_depth_ann = rep.annotators.get("distance_to_image_plane")
        self._global_depth_ann.attach([global_rp])

        # -- 2. Wrist Camera (Mounted on ee_link) --------------------------------
        # Define UsdGeom.Camera directly under ee_link to inherit EE world transforms.
        wrist_cam_path = "/World/Franka/panda_link7/WristCamera"
        stage = self._omni_usd.get_context().get_stage()
        usd_cam = self._UsdGeom.Camera.Define(stage, wrist_cam_path)
        usd_cam.GetFocalLengthAttr().Set(8.0)
        usd_cam.GetHorizontalApertureAttr().Set(24.0)
        usd_cam.GetClippingRangeAttr().Set(self._Gf.Vec2f(0.01, WRIST_DEPTH_MAX))

        # Local offset and orientation relative to ee_link
        # USD cameras look down -Z by default; rotate 180 deg around X to align with EE Z+ forward.
        xformable = self._UsdGeom.Xformable(usd_cam.GetPrim())
        xformable.ClearXformOpOrder()
        ox, oy, oz = cfg.wrist_cam_offset
        xformable.AddTranslateOp().Set(self._Gf.Vec3d(ox, oy, oz))
        xformable.AddRotateXYZOp().Set(self._Gf.Vec3d(180.0, 0.0, 0.0))

        wrist_rp = rep.create.render_product(
            wrist_cam_path, (IMG_W, IMG_H), name="WristView", force_new=True)
        self._wrist_depth_ann = rep.annotators.get("distance_to_image_plane")
        self._wrist_depth_ann.attach([wrist_rp])

 # --------------------------------------------------------------------------
    # Public Interface: reset
    # --------------------------------------------------------------------------
    def reset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Resets the environment to a new episode starting point.
        Randomizes: hole position, target position, and initial joint angles (with small noise).

        Returns
        -------
        scalar_obs   : (SCALAR_DIM,) float32   Standard scalar observations
        priv_obs     : (PRIVILEGED_DIM,) float32 Privileged observations for Critic
        wrist_depth  : (IMG_H, IMG_W) float32  Depth map from wrist camera (meters)
        global_depth : (IMG_H, IMG_W) float32  Depth map from global camera (meters)
        info         : dict
        """
        cfg = self.cfg

        # -- Apply curriculum stage change request if pending ------------------
        if self._pending_stage is not None:
            if self._pending_stage != self.cfg.stage:
                self.set_stage(self._pending_stage)
            self._pending_stage = None

        # -- Reset counters and status flags -----------------------------------
        self._step_count = 0
        self._passed_board = False
        self._prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._episode_count += 1

        # -- Randomize Hole and Target -----------------------------------------
        hole_y = float(np.random.uniform(*cfg.hole_y_range))
        hole_z = float(np.random.uniform(*cfg.hole_z_range))
        # Hole center X is fixed at the board plane
        self._hole_center = np.array(
            [cfg.board_x, hole_y, hole_z], dtype=np.float32)

        target_x = float(np.random.uniform(*cfg.target_x_range))
        # Target Y/Z are randomized near the hole center for task continuity
        target_y = hole_y + float(np.random.uniform(*cfg.target_y_range))
        target_z = hole_z + float(np.random.uniform(*cfg.target_z_range))
        self._target_pos = np.array(
            [target_x, target_y, target_z], dtype=np.float32)

        # -- Update Board Geometry ---------------------------------------------
        self._reset_board(hole_y, hole_z)

        # -- Update Target Marker Position -------------------------------------
        self._target_marker.set_world_pose(
            position=self._target_pos,
        )

        # -- Reset Robot Joints to Initial Pose (with small noise for diversity) -
        init_q = np.array(cfg.init_joints, dtype=np.float32)
        noise = np.zeros(N_JOINTS_TOTAL, dtype=np.float32)
        _noise_scale = {1: 0.05, 2: 0.02,
                        3: 0.05, 4: 0.08}.get(cfg.stage, 0.05)
        noise[ACTIVE_JOINT_IDX] = np.random.uniform(
            -_noise_scale, _noise_scale, len(ACTIVE_JOINT_IDX))
        init_q_noisy = init_q + noise

        # Two steps required for physical teleportation:
        # 1. set_joint_positions: Instant state update (angles, velocities)
        # 2. apply_action: Set PD target to maintain the position in the next step
        self._robot.set_joint_positions(init_q_noisy)
        self._robot.set_joint_velocities(
            np.zeros(N_JOINTS_TOTAL, dtype=np.float32))
        self._robot.apply_action(self._ArtAction(joint_positions=init_q_noisy))

        # -- Advance simulation steps for stability ----------------------------
        # PD controllers need a few frames to converge to the target position
        for _ in range(5):
            self.world.step(render=False)
        # Final step triggers rendering for depth update
        self.world.step(render=True)

        # -- Log initial EE state ----------------------------------------------
        ee_pos, _ = self._get_ee_pose()
        self._prev_ee_x = float(ee_pos[0])
        self._prev_ee_pos = ee_pos.copy()   # For velocity estimation in step()

        # -- Reset Reward Manager (Initialize potential baselines) -------------
        self._reward_manager.reset(
            ee_pos=ee_pos,
            hole_center=self._hole_center,
            target_pos=self._target_pos,
        )

        if cfg.debug_diagnostics and self._episode_count <= cfg.debug_log_episodes:
            collided = self._check_collision(ee_pos)
            done, term_info = self._term_manager.check(
                step_count=0,
                ee_pos=ee_pos,
                passed_board=self._passed_board,
                target_pos=self._target_pos,
                collided=collided,
            )
            print(
                "[debug][reset] ep=%d ee_pos=%s init_q=%s collided=%s done=%s term=%s"
                % (self._episode_count, np.round(ee_pos, 3), np.round(init_q_noisy, 3),
                   collided, done, term_info)
            )

        # -- Get initial observations ------------------------------------------
        scalar_obs = self.scalar_norm(
            self._get_scalar_obs(), update=self.training)
        priv_obs = self.priv_norm(self._get_critic_obs(), update=self.training)
        wrist_depth = self._get_wrist_depth()
        global_depth = self._get_global_depth()

        info = {
            "hole_center": self._hole_center.copy(),
            "target_pos":  self._target_pos.copy(),
        }

        return scalar_obs, priv_obs, wrist_depth, global_depth, info

    # --------------------------------------------------------------------------
    # Public Interface: step
    # --------------------------------------------------------------------------
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, bool, dict]:
        """
        Executes one environment step.

        Parameters
        ----------
        action : (ACTION_DIM,) float32, range [-1, 1]
                 Passed directly from network output; no external scaling needed.

        Returns
        -------
        scalar_obs   : (SCALAR_DIM,) float32
        priv_obs     : (PRIVILEGED_DIM,) float32
        wrist_depth  : (IMG_H, IMG_W) float32
        global_depth : (IMG_H, IMG_W) float32
        reward       : float
        done         : bool
        info         : dict
        """
        cfg = self.cfg
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # -- Calculate joint targets (current pose + delta) ---------------------
        current_q = self._robot.get_joint_positions()
        # 6-DOF: action maps to joints 0-5; joint7 (index 6) remains fixed
        delta_q = np.zeros(N_JOINTS_TOTAL, dtype=np.float32)
        delta_q[ACTIVE_JOINT_IDX] = action * cfg.action_scale
        target_q = current_q + delta_q

        # -- Send joint position commands ---------------------------------------
        self._robot.apply_action(
            self._ArtAction(joint_positions=target_q)
        )

        # -- Advance physics ---------------------------------------------------
        self.world.step(render=True)
        self._step_count += 1

        # -- Get new state (using PhysX tensor API for consistency) ------------
        ee_pos, ee_quat = self._get_ee_pose()

        # -- EE Velocity (Estimated via finite difference for R_align/Critic) --
        ee_vel = (ee_pos - self._prev_ee_pos) / cfg.physics_dt

        # -- Collision / Piercing Detection -------------------------------------
        collided = self._check_collision(ee_pos)
        just_passed = self._check_pass_through(ee_pos)

        # Retreat penalty: prevents the strategy of passing through and then exiting
        just_retreated = (self._passed_board and
                          self._prev_ee_x >= self.cfg.board_x and
                          ee_pos[0] < self.cfg.board_x)
        reward_retreat = -3.0 if just_retreated else 0.0

        # -- Compute Reward ----------------------------------------------------
        reward = self._reward_manager.compute(
            ee_pos=ee_pos,
            ee_vel=ee_vel,
            action=action,
            prev_action=self._prev_action,
            hole_center=self._hole_center,
            target_pos=self._target_pos,
            passed_board=self._passed_board,
            collided=collided,
            just_passed=just_passed,
        ) + reward_retreat

        # -- Check Termination (delegated to TerminationManager) ---------------
        done, term_info = self._term_manager.check(
            step_count=self._step_count,
            ee_pos=ee_pos,
            passed_board=self._passed_board,
            target_pos=self._target_pos,
            collided=collided,
        )

        # -- Get Observations --------------------------------------------------
        scalar_obs = self.scalar_norm(
            self._get_scalar_obs(), update=self.training)
        priv_obs = self.priv_norm(self._get_critic_obs(), update=self.training)
        wrist_depth = self._get_wrist_depth()
        global_depth = self._get_global_depth()

        info = {**term_info, "ee_pos": ee_pos.copy(), "stage": self.cfg.stage}

        # -- Automatic Curriculum Switching: Based on rolling window stats ------
        if done and self.training:
            pass_success = 1.0 if self._passed_board else 0.0
            arrive_success = 1.0 if bool(
                term_info.get("success", False)) else 0.0
            self._pass_history.append(pass_success)
            self._arrive_history.append(arrive_success)

            if len(self._pass_history) == self._curriculum_window:
                pass_rate_100 = float(np.mean(self._pass_history))
                arrive_rate_100 = float(np.mean(self._arrive_history))
                info["pass_rate_rolling"] = pass_rate_100
                info["arrive_rate_rolling"] = arrive_rate_100

                # Curriculum transition logic
                if self.cfg.stage == 1 and pass_rate_100 >= 0.6:
                    if self._pending_stage is None:
                        self.set_curriculum_stage(2)
                elif self.cfg.stage == 2 and pass_rate_100 >= 0.8 and arrive_rate_100 >= 0.7:
                    if self._pending_stage is None:
                        self.set_curriculum_stage(3)
                elif self.cfg.stage == 3 and pass_rate_100 >= 0.95 and arrive_rate_100 >= 0.8:
                    if self._pending_stage is None:
                        self.set_curriculum_stage(4)

        if cfg.debug_diagnostics and self._episode_count <= cfg.debug_log_episodes:
            if self._step_count <= cfg.debug_log_steps or done:
                bd = self._reward_manager.last_breakdown
                q_after = self._robot.get_joint_positions()
                q_move = float(
                    np.mean(np.abs(q_after[ACTIVE_JOINT_IDX] - current_q[ACTIVE_JOINT_IDX])))
                q_err = float(
                    np.mean(np.abs(target_q[ACTIVE_JOINT_IDX] - q_after[ACTIVE_JOINT_IDX])))
                ee_delta = float(np.linalg.norm(ee_pos - self._prev_ee_pos))
                print(
                    "[debug][step] ep=%d step=%d q_move=%.4f q_err=%.4f ee_delta=%.4f reward=%.4f done=%s term=%s bd=%s"
                    % (self._episode_count, self._step_count, q_move, q_err, ee_delta,
                       float(reward), done, term_info, {k: round(float(v), 4) for k, v in bd.items()})
                )

        # -- Update state cache ------------------------------------------------
        self._prev_action = action.copy()
        self._prev_ee_x = float(ee_pos[0])
        self._prev_ee_pos = ee_pos.copy()

        return scalar_obs, priv_obs, wrist_depth, global_depth, reward, done, info

    # --------------------------------------------------------------------------
    # Public Interface: close
    # --------------------------------------------------------------------------

    def close(self):
        """
        Shuts down the Isaac Sim simulation. Called by train.py after training.
        Must stop the replicator orchestrator before closing the app to prevent 
        crashes in omni.graph.core during atexit.
        """
        try:
            self._rep.orchestrator.stop()
        except Exception:
            pass
        self._sim_app.close()

    # --------------------------------------------------------------------------
    # Internal: Board Geometry Randomization
    # --------------------------------------------------------------------------
    def _reset_board(self, hole_y: float, hole_z: float):
        """
        Recalculates positions and scales for the 4 board pieces (Top/Bottom/Left/Right)
        every episode to create a square hole at a randomized location.

        Board Layout and Naming:

            ┌────────── [Top] ──────────┐   z_high
            │                           │
          [Left]      [HOLE]         [Right]
            │                           │
            └─────── [Bottom] ──────────┘   z_low
           y_min   hole_y-h  hole_y+h  y_max

        Where h = cfg.hole_half_size

        Parameters
        ----------
        hole_y : Hole center Y coordinate (Board plane frame)
        hole_z : Hole center Z coordinate (Board plane frame)
        """
        cfg = self.cfg
        bx = cfg.board_x        # Board X position
        th = cfg.board_thickness
        h = cfg.hole_half_size
        y0 = -cfg.board_half_y   # Board left edge Y
        y1 = cfg.board_half_y    # Board right edge Y
        z0 = cfg.board_z_low     # Board bottom edge Z
        z1 = cfg.board_z_high    # Board top edge Z

        # Hole boundaries (Clipped to ensure hole stays within board limits)
        hy_lo = np.clip(hole_y - h, y0 + 0.01, y1 - 0.01)
        hy_hi = np.clip(hole_y + h, y0 + 0.01, y1 - 0.01)
        hz_lo = np.clip(hole_z - h, z0 + 0.01, z1 - 0.01)
        hz_hi = np.clip(hole_z + h, z0 + 0.01, z1 - 0.01)

        # Geometry for each piece: (center_y, center_z, size_y, size_z)
        pieces_geom = {
            "Bottom": (
                (y0 + y1) / 2,           (z0 + hz_lo) / 2,
                y1 - y0,                  hz_lo - z0,
            ),
            "Top": (
                (y0 + y1) / 2,           (hz_hi + z1) / 2,
                y1 - y0,                  z1 - hz_hi,
            ),
            "Left": (
                (y0 + hy_lo) / 2,        (hz_lo + hz_hi) / 2,
                hy_lo - y0,               hz_hi - hz_lo,
            ),
            "Right": (
                (hy_hi + y1) / 2,        (hz_lo + hz_hi) / 2,
                y1 - hy_hi,               hz_hi - hz_lo,
            ),
        }

        names = ["Top", "Bottom", "Left", "Right"]
        for piece, name in zip(self._board_pieces, names):
            cy, cz, sy, sz = pieces_geom[name]
            # Thickness is along X axis; Y/Z are determined by hole position
            piece.set_world_pose(
                position=np.array([bx, cy, cz])
            )
            piece.set_local_scale(
                np.array([th, max(sy, 0.01), max(sz, 0.01)])
            )

    # --------------------------------------------------------------------------
    # Internal: End-Effector Pose Retrieval
    # --------------------------------------------------------------------------
    def _get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gets Franka EE (panda_hand) world position and quaternion.

        Returns
        -------
        ee_pos  : (3,) float32 [x, y, z] (meters)
        ee_quat : (4,) float32 [w, x, y, z] (Isaac Sim convention)
        """
        # Lazy Init: PhysX Articulation View requires world.step() to have been called
        if self._art_physx_view is None:
            self._init_physx_art_view()

        # Native PhysX Tensor API for high-performance state retrieval
        self._physics_sim_view.update_articulations_kinematic()
        t = self._art_physx_view.get_link_transforms()[
            0, self._ee_link_physx_idx]

        ee_pos = np.array([t[0], t[1], t[2]], dtype=np.float32)
        # PhysX: [qx, qy, qz, qw] -> Isaac Sim: [qw, qx, qy, qz]
        ee_quat = np.array([t[6], t[3], t[4], t[5]], dtype=np.float32)
        return ee_pos, ee_quat

    def _init_physx_art_view(self):
        """
        Initializes PhysX tensor views. Reuses existing views from robot object.
        Determines link indices based on actual PhysX metadata.
        """
        self._art_physx_view = self._robot._articulation_view._physics_view
        self._physics_sim_view = self._SimulationManager.get_physics_sim_view()

        physx_link_names = list(
            self._art_physx_view.shared_metatype.link_names)
        print(
            f"[HoleBoardEnv] PhysX links ({len(physx_link_names)}): {physx_link_names}")

        # Map EE link index: Prefer 'panda_hand', fallback to 'panda_link8'
        if "panda_hand" in physx_link_names:
            self._ee_link_physx_idx = physx_link_names.index("panda_hand")
        elif "panda_link8" in physx_link_names:
            self._ee_link_physx_idx = physx_link_names.index("panda_link8")
        else:
            raise RuntimeError(
                f"Franka EE link not found. Available links: {physx_link_names}")

        # Indices for arm links (used in privileged observations)
        _arm_link_names = ["panda_link1", "panda_link2", "panda_link3",
                           "panda_link4", "panda_link5", "panda_link6"]
        self._arm_link_physx_idxs = [physx_link_names.index(
            n) for n in _arm_link_names if n in physx_link_names]

    # --------------------------------------------------------------------------
    # Internal: Scalar Observation Vector Construction
    # --------------------------------------------------------------------------
    def _get_scalar_obs(self) -> np.ndarray:
        """
        Constructs Actor scalar observations (SCALAR_DIM = 31):

        [0:6]   sin-encoded joint angles (joint1-6)
        [6:12]  cos-encoded joint angles
                - Encoding as (sin, cos) prevents discontinuities at ±π
        [12:18] joint velocities (rad/s)
        [18:21] EE position (x, y, z)
        [21:25] EE quaternion (w, x, y, z)
        [25:28] Unit vector EE -> Target (3D)
        [28]    Euclidean distance EE -> Target (m)
        [29]    Piercing flag passed_board ∈ {0, 1}
        [30]    Survival time step ratio step_count/max_steps ∈ [0,1]

        Note: Actor does NOT see EE -> Hole information; it must rely on CNN features.

        Returns
        -------
        obs : (31,) float32
        """
        q = self._robot.get_joint_positions().astype(np.float32)
        qdot = self._robot.get_joint_velocities().astype(np.float32)

        q_act = q[ACTIVE_JOINT_IDX]
        qdot_act = qdot[ACTIVE_JOINT_IDX]

        sin_q, cos_q = np.sin(q_act), np.cos(q_act)
        ee_pos, ee_quat = self._get_ee_pose()

        vec_to_target = self._target_pos - ee_pos
        dist_to_target = float(np.linalg.norm(vec_to_target))
        unit_to_target = vec_to_target / (dist_to_target + 1e-8)

        alive_t = np.array(
            [self._step_count / self.cfg.max_steps], dtype=np.float32)

        obs = np.concatenate([
            sin_q, cos_q, qdot_act, ee_pos, ee_quat,
            unit_to_target, [dist_to_target], [
                float(self._passed_board)], alive_t
        ], dtype=np.float32)

        assert obs.shape == (SCALAR_DIM,), f"obs shape mismatch: {obs.shape}"
        return obs

    # --------------------------------------------------------------------------
    # Internal: Privileged (Critic) Observation Vector Construction
    # --------------------------------------------------------------------------
    def _get_critic_obs(self) -> np.ndarray:
        """
        Constructs Critic privileged observations (PRIVILEGED_DIM = 48):
        Pure scalar state directly read from simulation (No CNN involved).

        [0:18]  Joint states (sin, cos, qdot)
        [18:25] EE pose (pos, quat)
        [25:28] Hole center world coordinates
        [28:31] Hole normal vector (fixed to [1, 0, 0])
        [31:34] Raw vector EE -> Hole
        [34:37] Raw vector EE -> Target
        [37:43] Signed distance from 6 arm links to board plane (m)
        [43]    cos(θ) between EE tool axis and hole normal
        [44:47] Board center world coordinates
        [47]    Hole diameter (2 * hole_half_size)

        Returns
        -------
        critic_obs : (48,) float32
        """
        cfg = self.cfg
        q = self._robot.get_joint_positions().astype(np.float32)
        qdot = self._robot.get_joint_velocities().astype(np.float32)

        q_act, qdot_act = q[ACTIVE_JOINT_IDX], qdot[ACTIVE_JOINT_IDX]
        sin_q, cos_q = np.sin(q_act), np.cos(q_act)
        ee_pos, ee_quat = self._get_ee_pose()

        hole_center = self._hole_center.copy()
        hole_normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        vec_ee_to_hole = hole_center - ee_pos
        vec_ee_to_target = self._target_pos - ee_pos

        # Distance from arm links to board plane
        joint_dists = np.zeros(len(ACTIVE_JOINT_IDX), dtype=np.float32)
        link_transforms = self._art_physx_view.get_link_transforms()
        for i, idx in enumerate(self._arm_link_physx_idxs):
            joint_dists[i] = float(link_transforms[0, idx, 0]) - cfg.board_x

        # cos(θ) between EE tool axis (local Z) and hole normal [1, 0, 0]
        w, x, y, z = ee_quat
        ee_fwd_x = 2.0 * (x * z + w * y)
        cos_axis_angle = float(np.clip(ee_fwd_x, -1.0, 1.0))

        board_center = np.array(
            [cfg.board_x, 0.0, (cfg.board_z_low + cfg.board_z_high) / 2.0], dtype=np.float32)
        hole_diam = np.array([2.0 * cfg.hole_half_size], dtype=np.float32)

        critic_obs = np.concatenate([
            sin_q, cos_q, qdot_act, ee_pos, ee_quat,
            hole_center, hole_normal, vec_ee_to_hole, vec_ee_to_target,
            joint_dists, [cos_axis_angle], board_center, hole_diam
        ], dtype=np.float32)

        assert critic_obs.shape == (
            PRIVILEGED_DIM,), f"critic_obs mismatch: {critic_obs.shape}"
        return critic_obs

# --------------------------------------------------------------------------
    # Internal: Depth Map Retrieval and Normalization
    # --------------------------------------------------------------------------
    def _normalize_depth(self, annotator, max_depth: float, label: str) -> np.ndarray:
        """
        Reads raw depth from replicator annotator, clips and normalizes to [0, 1].

        Parameters
        ----------
        annotator : replicator annotator object
        max_depth : Clip upper bound (meters), corresponding to 1.0 after normalization
        label     : Label for WARN prints to distinguish wrist/global cameras

        Returns
        -------
        depth_norm : (IMG_H, IMG_W) float32, range [0, 1]
        """
        try:
            raw = annotator.get_data()
            if raw is None or raw.size == 0:
                return np.zeros((IMG_H, IMG_W), dtype=np.float32)

            # Handle multi-channel depth if present
            if raw.ndim == 3:
                raw = raw[:, :, 0]

            depth = raw.astype(np.float32)
            # Replace invalid values (NaN, Inf) with bounds
            depth = np.nan_to_num(depth, nan=max_depth,
                                  posinf=max_depth, neginf=0.0)

            # Clip and normalize to [0, 1]
            depth = np.clip(depth, 0.0, max_depth) / max_depth
        except Exception as e:
            print(
                f"[WARN] Failed to retrieve {label} depth map, returning zero map: {e}")
            depth = np.zeros((IMG_H, IMG_W), dtype=np.float32)
        return depth

    def _get_wrist_depth(self) -> np.ndarray:
        """Wrist camera depth map, normalized to [0, 1] with WRIST_DEPTH_MAX clip."""
        return self._normalize_depth(self._wrist_depth_ann, WRIST_DEPTH_MAX, "wrist")

    def _get_global_depth(self) -> np.ndarray:
        """Global camera depth map, normalized to [0, 1] with GLOBAL_DEPTH_MAX clip."""
        return self._normalize_depth(self._global_depth_ann, GLOBAL_DEPTH_MAX, "global")

    # --------------------------------------------------------------------------
    # Public Interface: Curriculum Stage Management
    # --------------------------------------------------------------------------
    # Geometry parameter overrides for each curriculum stage.
    # Stages not listed here use default values from EnvConfig.
    _STAGE_GEOM = {
        # Stage 1: Uses default EnvConfig geometry
        # Stage 2: Relaxed success_dist + target placed directly behind the hole
        2: dict(success_dist=0.06,
                target_x_range=(0.35, 0.4),
                target_y_range=(-0.05, 0.05),
                target_z_range=(-0.05, 0.05)),
        # Stage 3: Bridge stage; moderate target offset expansion
        3: dict(success_dist=0.05,
                hole_y_range=(-0.15, 0.15),
                target_x_range=(0.35, 0.45),
                target_y_range=(-0.1, 0.1),
                target_z_range=(-0.1, 0.1)),
        # Stage 4: Increased randomization + tightened success criterion
        4: dict(
            hole_half_size=0.16,
            hole_y_range=(-0.15,  0.15),
            hole_z_range=(0.5,  0.60),
            target_x_range=(0.35, 0.5),
            target_y_range=(-0.1, 0.1),
            target_z_range=(-0.1, 0.1),
            success_dist=0.04,
        ),
    }

    def set_stage(self, stage: int):
        """
        Switches the curriculum stage (1/2/3/4). 
        Updates RewardManager weights and scene geometry configurations accordingly.
        """
        assert stage in (1, 2, 3, 4), f"Stage must be 1/2/3/4, got {stage}"
        self.cfg.stage = stage
        self._reward_manager.set_stage(stage)

        # Clear rolling windows to prevent historical data from affecting new stage transitions
        self._pass_history.clear()
        self._arrive_history.clear()

        # Apply per-stage geometry overrides (Reset to defaults before overriding)
        self.cfg.hole_half_size = 0.18
        self.cfg.hole_y_range = (-0.1, 0.1)
        self.cfg.hole_z_range = (0.55, 0.58)
        self.cfg.target_y_range = (-0.1, 0.1)
        self.cfg.target_z_range = (-0.1, 0.1)
        self.cfg.success_dist = 0.05

        geom = self._STAGE_GEOM.get(stage, {})
        for k, v in geom.items():
            setattr(self.cfg, k, v)

        # Sync success_dist to TerminationManager
        self._term_manager._success_dist = self.cfg.success_dist

        print(f"[stage] Switched to {stage} | hole_half={self.cfg.hole_half_size:.3f}m "
              f"target_y={self.cfg.target_y_range} success_dist={self.cfg.success_dist:.3f}m")

    def set_curriculum_stage(self, stage: int):
        """
        Requests a curriculum switch to a specific stage. 
        The switch takes effect at the beginning of the next reset.
        """
        assert stage in (1, 2, 3, 4), f"Stage must be 1/2/3/4, got {stage}"
        if self._pending_stage == stage:
            return
        self._pending_stage = stage
        print(f"[curriculum] Queued stage {stage} (apply on next reset)")

    @property
    def reward_breakdown(self) -> dict:
        """Returns the breakdown of the last step's reward terms for evaluation/logging."""
        return self._reward_manager.last_breakdown

    # --------------------------------------------------------------------------
    # Internal: Collision Detection (Geometric Method)
    # --------------------------------------------------------------------------
    def _check_collision(self, ee_pos: np.ndarray) -> bool:
        """
        Detects collisions between the EE and the board using geometric checks.

        Logic:
          1. EE's X-coordinate is within the board's thickness (crossing the plane).
          2. EE's Y/Z coordinates fall within the solid area of the board (not in the hole).

        This method is more stable than physical contact sensors (independent of 
        physics time-steps) and sufficiently accurate for this task.

        Parameters
        ----------
        ee_pos : (3,) Current EE world coordinates

        Returns
        -------
        bool : True if a collision is detected
        """
        cfg = self.cfg
        bx = cfg.board_x
        th = cfg.board_thickness / 2.0

        # Condition 1: EE X-coordinate is within the board thickness range
        x_in_board = (bx - th) < ee_pos[0] < (bx + th)
        if not x_in_board:
            return False

        # Condition 2: EE Y/Z is within the overall board dimensions
        y_in_board = -cfg.board_half_y < ee_pos[1] < cfg.board_half_y
        z_in_board = cfg.board_z_low < ee_pos[2] < cfg.board_z_high
        if not (y_in_board and z_in_board):
            return False

        # Condition 3: Check if EE is within the hole boundaries
        h = cfg.hole_half_size
        hy, hz = self._hole_center[1], self._hole_center[2]
        in_hole = (abs(ee_pos[1] - hy) < h) and (abs(ee_pos[2] - hz) < h)

        return not in_hole   # In solid area of the board = Collision

    # --------------------------------------------------------------------------
    # Internal: Piercing (Pass-through) Detection
    # --------------------------------------------------------------------------
    def _check_pass_through(self, ee_pos: np.ndarray) -> bool:
        """
        Detects if the EE has just successfully pierced the board.

        Conditions:
          - Previous step EE was in front of the board (prev_x < board_x).
          - Current step EE is behind the board (curr_x > board_x).
          - During transition, the EE Y/Z coordinates were within the hole boundaries.

        Note: _check_collision is evaluated before this function; a collision 
        terminates the episode, so piercing won't trigger on a collision frame.

        Returns
        -------
        bool : True if piercing was just completed
        """
        cfg = self.cfg
        bx = cfg.board_x

        if self._passed_board:
            return False   # Already passed, do not trigger again

        just_crossed = self._prev_ee_x < bx <= ee_pos[0]
        if not just_crossed:
            return False

        # Verify that the EE was within the hole during the crossover
        h = cfg.hole_half_size
        hy, hz = self._hole_center[1], self._hole_center[2]
        in_hole = (abs(ee_pos[1] - hy) < h) and (abs(ee_pos[2] - hz) < h)

        if in_hole:
            self._passed_board = True
            return True

        return False

    # --------------------------------------------------------------------------
    # Properties: For dimension queries in ppo.py / network.py
    # --------------------------------------------------------------------------
    @property
    def scalar_obs_dim(self) -> int:
        """Dimension of scalar observations for policy network construction."""
        return SCALAR_DIM

    @property
    def depth_img_shape(self) -> Tuple[int, int]:
        """Resolution (H, W) of the depth map for CNN construction."""
        return (IMG_H, IMG_W)

    @property
    def action_dim(self) -> int:
        """Dimension of the action space."""
        return ACTION_DIM


# ------------------------------------------------------------------------------
# Independent Test Entry Point (Run 'python env.py' directly to verify setup)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    print("=" * 50)
    print("  env.py Standalone Test Mode")
    print("  Verifying: Scene construction / reset / step interfaces")
    print("  Close the window or press Ctrl+C to exit")
    print("=" * 50)

    # SimulationApp must be created at the top-most scope and as the first statement
    sim_app = SimulationApp(
        {"headless": False, "renderer": "RayTracedLighting"})

    cfg = EnvConfig(headless=False, max_steps=200)
    env = HoleBoardEnv(cfg, sim_app=sim_app)

    try:
        # -- First Episode: Print observation dimensions -----------------------
        scalar_obs, priv_obs, wrist_depth, global_depth, info = env.reset()
        print(
            f"\n[reset] scalar_obs shape  : {scalar_obs.shape}   Expected: ({SCALAR_DIM},)")
        print(
            f"[reset] priv_obs shape    : {priv_obs.shape}   Expected: ({PRIVILEGED_DIM},)")
        print(
            f"[reset] wrist_depth shape : {wrist_depth.shape}   Expected: ({IMG_H}, {IMG_W})")
        print(
            f"[reset] global_depth shape: {global_depth.shape}   Expected: ({IMG_H}, {IMG_W})")
        print(f"[reset] hole_center       : {info['hole_center']}")
        print(f"[reset] target_pos        : {info['target_pos']}")
        print("\nWindow opened. Standing still... Close window to exit.\n")

        # -- Continuous Loop: Automatic reset after each episode to keep window visible --
        episode = 0
        while sim_app.is_running():
            # Action set to zero for now to keep the arm still
            action = np.zeros(ACTION_DIM, dtype=np.float32)
            # action = np.random.uniform(-1.0, 1.0, ACTION_DIM).astype(np.float32)

            obs, priv, wrist_d, global_d, reward, done, info = env.step(action)

            if done:
                episode += 1
                result = ("SUCCESS  " if info["success"]
                          else "COLLISION" if info["collision"]
                          else "TIMEOUT  ")
                print(f"[ep {episode:3d}]  {result}  "
                      f"steps={info['step_count']:3d}  "
                      f"dist_to_target={info['dist_to_target']:.3f}m")

                # Reset for the next episode
                obs, priv, wrist_d, global_d, info = env.reset()

    except KeyboardInterrupt:
        print("\n[test] User interrupted.")
    finally:
        # Must stop replicator before closing the app to prevent omni.graph crashes
        env.close()
        # Force exit to skip remaining Python atexit cleanup (already handled by Isaac Sim)
        sys.exit(0)
