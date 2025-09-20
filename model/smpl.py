import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import smplx
import math
import cv2

def recursive_unique_assignment(
    splats,
    references,
    k=2,
    device="cuda",
    used_refs=None,
    splat_indices=None,
    final_assignments=None,
    depth=0,
):
    """
    Recursive greedy assignment of splats to unique reference points.

    Args:
        splats (Tensor): (M, 3) unassigned splat coordinates
        references (Tensor): (N, 3) all reference points
        k (int): top-k nearest unassigned refs to consider
        device (str): 'cuda' or 'cpu'
        used_refs (set): indices of references already assigned
        splat_indices (Tensor): (M,) global indices of current splats
        final_assignments (Tensor): (M_total,) to fill with ref indices
        depth (int): recursion depth (for logging/debugging)

    Returns:
        final_assignments (Tensor): (M_total,) — reference index per splat
    """
    splats = splats.to(device)
    references = references.to(device)
    M, N = splats.shape[0], references.shape[0]

    if used_refs is None:
        used_refs = set()

    if splat_indices is None:
        splat_indices = torch.arange(M, device=device)

    if final_assignments is None:
        M_total = splat_indices.max().item() + 1
        final_assignments = -torch.ones(M_total, dtype=torch.long, device='cpu')

    # --- Filter references to only unused ones ---
    available_ref_indices = torch.tensor(
        sorted(set(range(N)) - used_refs),
        dtype=torch.long,
        device=device
    )
    available_refs = references[available_ref_indices]  # (R, 3)

    if available_refs.shape[0] == 0:
        raise RuntimeError("No available references left for assignment.")

    # --- Compute distances to available refs only ---
    with torch.no_grad():
        dists = torch.cdist(splats, available_refs)  # (M, R)

    # --- Get top-k nearest unassigned references ---
    k_eff = min(k, available_refs.shape[0])  # avoid overflow
    knn_dists, knn_indices = torch.topk(dists, k=k_eff, largest=False, dim=1)

    local_used = set()

    for i in range(M):
        for j in range(k_eff):
            local_idx = knn_indices[i, j].item()
            global_ref = available_ref_indices[local_idx].item()
            if global_ref not in used_refs and global_ref not in local_used:
                final_assignments[splat_indices[i].item()] = global_ref
                local_used.add(global_ref)
                break

    used_refs.update(local_used)

    # --- Recurse on unassigned splats ---
    unassigned_mask = torch.tensor([
        final_assignments[splat_indices[i].item()] == -1 for i in range(M)
    ], dtype=torch.bool)

    if unassigned_mask.any():
        unassigned_splats = splats[unassigned_mask]
        unassigned_indices = splat_indices[unassigned_mask]

        print(f"[Depth {depth}] Recursing on {unassigned_splats.shape[0]} unassigned splats...")

        recursive_unique_assignment(
            unassigned_splats,
            references,
            k=k,
            device=device,
            used_refs=used_refs,
            splat_indices=unassigned_indices,
            final_assignments=final_assignments,
            depth=depth + 1,
        )

    return final_assignments


def invert_assignments(assignments: torch.Tensor, num_references: int) -> torch.Tensor:
    """
    Create a reverse mapping from reference point to splat index.

    Args:
        assignments (Tensor): (M,) tensor of assigned reference indices
        num_references (int): total number of reference points (N)

    Returns:
        reverse_assignments (Tensor): (N,) where each entry is splat index or -1
    """
    reverse = -torch.ones(num_references, dtype=torch.long, device=assignments.device)

    reverse[assignments] = torch.arange(
        len(assignments), dtype=torch.long, device=assignments.device
    )

    return reverse


# -----------------------------------------------------
# Helpers: normals, per-vertex & per-face Gaussians
# -----------------------------------------------------
def compute_vertex_normals(vertices_np, faces_np):
    v = torch.tensor(vertices_np, dtype=torch.float32)
    f = torch.tensor(faces_np, dtype=torch.long)
    normals = torch.zeros_like(v)
    tris = v[f]  # (F,3,3)
    tri_normals = torch.linalg.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    for i in range(3):
        normals.index_add_(0, f[:, i], tri_normals)
    normals = torch.nn.functional.normalize(normals, dim=1).cpu().numpy()
    return normals


def build_vertex_gaussians(vertices_np, faces_np, min_scale=0.002, max_scale=0.05, device="cuda"):
    V = vertices_np.shape[0]
    # adjacency for average edge length
    neighbors = [[] for _ in range(V)]
    for f in faces_np:
        a, b, c = f
        neighbors[a] += [b, c]
        neighbors[b] += [a, c]
        neighbors[c] += [a, b]

    normals = compute_vertex_normals(vertices_np, faces_np)

    v_means, v_scales, v_quats = [], [], []
    for i in range(V):
        center = vertices_np[i]
        if neighbors[i]:
            neigh = vertices_np[neighbors[i]]
            avg_len = np.mean(np.linalg.norm(neigh - center, axis=1)) / 2
        else:
            avg_len = 0.005

        scale = np.array([avg_len, avg_len, avg_len * 0.2], dtype=np.float32)
        scale = np.clip(scale, min_scale, max_scale)

        # Build rotation frame: z = normal; construct orthonormal basis
        z = normals[i] / (np.linalg.norm(normals[i]) + 1e-9)
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(np.dot(z, tmp)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        y = np.cross(z, tmp)
        y /= (np.linalg.norm(y) + 1e-9)
        x = np.cross(y, z)
        Rmat = np.stack([x, y, z], axis=1)  # columns are basis vectors
        quat_xyzw = R.from_matrix(Rmat).as_quat().astype(np.float32)  # xyzw
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]

        v_means.append(center.astype(np.float32))
        v_scales.append(scale)
        v_quats.append(quat_wxyz)

    return (
        torch.tensor(np.vstack(v_means), device=device),
        torch.tensor(np.vstack(v_scales), device=device),
        torch.tensor(np.vstack(v_quats), device=device),
    )


def build_face_gaussians(
    vertices_np, faces_np, inflate_normal=0.002, min_scale=0.002, max_scale=0.05
):
    f_means, f_scales, f_quats = [], [], []
    for tri_idx in range(faces_np.shape[0]):
        tri = vertices_np[faces_np[tri_idx]]
        center = tri.mean(axis=0)

        X = tri - center
        cov = (X.T @ X) / max(len(tri) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        v0, v1, v2 = tri
        normal = np.cross(v1 - v0, v2 - v0)
        normal = normal / (np.linalg.norm(normal) + 1e-10)
        eigvecs[:, -1] = normal

        eigvecs[:, 0] = eigvecs[:, 0] / np.linalg.norm(eigvecs[:, 0])
        eigvecs[:, 1] = np.cross(normal, eigvecs[:, 0])
        eigvecs[:, 1] /= np.linalg.norm(eigvecs[:, 1])
        eigvecs[:, 0] = np.cross(eigvecs[:, 1], normal)

        eigvals = np.clip(eigvals, min_scale**2, max_scale**2)
        eigvals[-1] = max(eigvals[-1], inflate_normal**2)
        scale = np.sqrt(eigvals).astype(np.float32)

        if np.linalg.det(eigvecs) < 0:
            eigvecs[:, -1] *= -1

        quat_xyzw = R.from_matrix(eigvecs).as_quat().astype(np.float32)
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]

        f_means.append(center.astype(np.float32))
        f_scales.append(scale)
        f_quats.append(quat_wxyz)

    return (
        torch.tensor(np.vstack(f_means)),
        torch.tensor(np.vstack(f_scales)),
        torch.tensor(np.vstack(f_quats)),
    )


def quaternion_normalize(quaternions):
    """
    Normalize a batch of quaternions to unit quaternions.
    Args:
        quaternions: Tensor of shape (..., 4) representing quaternions in (w, x, y, z) format.
    Returns:
        Normalized quaternions of the same shape.
    """
    norms = torch.norm(quaternions, dim=-1, keepdim=True)
    return quaternions / norms

SMPL_TO_BODY_MAPPING = [
    24,  # Nose
    12,  # Neck
    16,  # RShoulder
    18,  # RElbow
    20,  # RWrist
    17,  # LShoulder
    19,  # LElbow
    21,  # LWrist
    # 0, # Pelvis
    1,  # RHip
    4,  # RKnee
    7,  # RAnkle
    2,  # LHip
    5,  # LKnee
    8,  # LAnkle
    26,  # LEye
    25,  # REye
    28,  # LEar
    27,  # REar
]

# OpenPose Body connection pairs for skeleton drawing
BODY_PAIRS = [
    [2, 3],
    [2, 6],
    [3, 4],
    [4, 5],
    [6, 7],
    [7, 8],
    [2, 9],
    [9, 10],
    [10, 11],
    [2, 12],
    [12, 13],
    [13, 14],
    [2, 1],
    [1, 15],
    [15, 17],
    [1, 16],
    [16, 18],
]
BODY_COLORS = [
    [255, 0, 0],
    [255, 85, 0],
    [255, 170, 0],
    [255, 255, 0],
    [170, 255, 0],
    [85, 255, 0],
    [0, 255, 0],
    [0, 255, 85],
    [0, 255, 170],
    [0, 255, 255],
    [0, 170, 255],
    [0, 85, 255],
    [0, 0, 255],
    [85, 0, 255],
    [170, 0, 255],
    [255, 0, 255],
    [255, 0, 170],
    [255, 0, 85],
]


def convert_smpl_to_body(smpl_joints):
    """
    Convert SMPL joints (45,3) to OpenPose Body format (18,3)

    Args:
        smpl_joints: numpy array of shape (45, 3) - SMPL joint positions

    Returns:
        body_joints: numpy array of shape (18, 3) - Body joint positions
    """
    # Initialize Body joints with NaN (missing joints)
    body_joints = smpl_joints[SMPL_TO_BODY_MAPPING]
    return body_joints


def determine_valid_joint_based_on_ndc(joints_ndc):
    """
    Determine self-occluded joints based on NDC z-values heuristically.
    Joints are indexed as:
    [0]Nose, [1]Neck, [2]RShoulder, [3]RElbow, [4]RWrist,
    [5]LShoulder, [6]LElbow, [7]LWrist,
    [8]RHip, [9]RKnee, [10]RAnkle,
    [11]LHip, [12]LKnee, [13]LAnkle,
    [14]LEye, [15]REye, [16]LEar, [17]REar
    """
    visible = np.ones(len(joints_ndc), dtype=bool)
    z = joints_ndc[:, 2]

    # Reference points
    neck_z = z[1]
    nose_z = z[0]

    # --- Face ---
    eye_mean_z = 1 / 2 * (z[14] + z[15])
    if eye_mean_z < nose_z:
        visible[14] = False  # LEye
        visible[15] = False  # REye
        visible[0] = False
    if z[16] < neck_z and neck_z < z[17]:
        visible[17] = False  # REar
    elif z[17] < neck_z and neck_z < z[16]:
        visible[16] = False  # LEar

    # --- Arms ---
    if z[2] < neck_z and neck_z < z[5]:
        if z[3] < neck_z and neck_z < z[6]:
            if z[4] < z[7]:
                visible[5] = False
                visible[6] = False
                visible[7] = False
    elif z[5] < neck_z and neck_z < z[2]:
        if z[6] < neck_z and neck_z < z[3]:
            if z[7] < z[4]:
                visible[2] = False
                visible[3] = False
                visible[4] = False

    return visible


def smpl_to_openpose(
    smpl_joints,
    full_proj_transform,
    image_width,
    image_height,
    draw_skeleton=True,
):
    """
    Convert SMPL joints to OpenPose v2 Body format with size-dependent thickness.
    """

    # Convert inputs to numpy
    if isinstance(smpl_joints, torch.Tensor):
        smpl_joints = smpl_joints.detach().cpu().numpy()
    if isinstance(full_proj_transform, torch.Tensor):
        full_proj_transform = full_proj_transform.detach().cpu().numpy()
    full_proj_transform = full_proj_transform.squeeze()

    # SMPL -> Body
    body_3d = convert_smpl_to_body(smpl_joints)

    # Project to 2D
    joints_h = np.concatenate([body_3d, np.ones((len(body_3d), 1))], axis=1)
    proj = joints_h @ full_proj_transform
    w_coords = np.where(np.abs(proj[:, 3:4]) < 1e-8, 1e-8, proj[:, 3:4])
    joints_ndc = proj[:, :3] / w_coords
    x_pix = (joints_ndc[:, 0] * 0.5 + 0.5) * image_width
    y_pix = (joints_ndc[:, 1] * 0.5 + 0.5) * image_height

    x_valid = np.logical_and(x_pix >= 0, x_pix < image_width)
    y_valid = np.logical_and(y_pix >= 0, y_pix < image_height)
    visible = np.logical_and(x_valid, y_valid)
    valid = visible  # np.logical_and(visible, determine_valid_joint_based_on_ndc(joints_ndc))
    # Full Body 2D coords
    body_2d = np.stack([x_pix, y_pix], axis=1)

    # Create image
    img = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    # Draw joints
    for i, color in enumerate(BODY_COLORS):
        if valid[i]:
            pt = body_2d[i].astype(int)
            cv2.circle(img, tuple(pt), 4, color, -1)
            # cv2.putText(
            #     img,
            #     str(SMPL_TO_BODY_MAPPING[i]),
            #     (pt[0] + 6, pt[1] - 6),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5,
            #     color,
            #     1,
            #     cv2.LINE_AA,
            # )
    if draw_skeleton:
        for (i, j), color in zip(BODY_PAIRS, BODY_COLORS):
            pt1 = body_2d[i - 1].astype(int)
            pt2 = body_2d[j - 1].astype(int)
            if valid[i - 1] and valid[j - 1]:
                mY = np.mean([pt1[0], pt2[0]])
                mX = np.mean([pt1[1], pt2[1]])
                length = ((pt1 - pt2) ** 2).sum() ** 0.5
                angle = math.degrees(math.atan2(pt1[1] - pt2[1], pt1[0] - pt2[0]))
                polygon = cv2.ellipse2Poly(
                    (int(mY), int(mX)), (int(length / 2), 4), int(angle), 0, 360, 1
                )
                cv2.fillConvexPoly(img, polygon, [int(float(c) * 0.6) for c in color])

    return img, body_2d


class SMPL:

    def __init__(
        self,
        model_path,
        global_orient=[math.pi / 2, 0.0, 0.0],
        transl=[0.0, 0.0, 0.0],
        betas=None,
        device="cuda",
    ):
        self.model = smplx.create(
            model_path,
            model_type="smpl",
            batch_size=1,
        ).to(device)
        self.device = device
        self.betas = (
            torch.zeros([1, 10], device=device) if betas is None else betas.to(device)
        )
        self.body_pose = torch.zeros([1, 69], device=device)  # 23*3 axis-angle
        self.global_orient = torch.tensor([global_orient], device=device)  # stand up
        self.transl = torch.tensor([transl], device=device)
        with torch.no_grad():
            out = self.model(
                betas=self.betas,
                body_pose=self.body_pose,
                global_orient=self.global_orient,
                transl=self.transl,
                return_verts=True,
                return_joints=True,
            )
        self.rest_vertices = out.vertices[0].detach().cpu().numpy()  # (6890, 3)
        self.joints = out.joints[0].detach()
        self.faces = self.model.faces.astype(np.int64)  # (13776, 3)
        self.lbs_weights = self.model.lbs_weights.to(device)  # (6890, 24)
        self.build_initial_vertex_gaussians()

    def build_initial_vertex_gaussians(self):
        means, scales, quats = build_vertex_gaussians(
            self.rest_vertices, self.faces, device=self.device
        )
        self.rest_means = means
        self.means = self.rest_means.clone()
        self.rest_colors = None
        self.colors = None
        self.rest_opacities = torch.ones_like(self.rest_means[:, :1])
        self.opacities = self.rest_opacities.clone()
        self.rest_scales = scales
        self.scales = self.rest_scales.clone()
        self.rest_quats = quats
        self.quats = self.rest_quats.clone()

    def apply_pose(
        self, body_pose=None, global_orient=None, transl=None, orthogonalize=True
    ):
        """
        Pose by linearly blending joint rotation matrices and translations, then transform
        per-vertex Gaussians (means + covariances).
        Args:
            body_pose: (1,69) axis-angle
            global_orient: (1,3)
            transl: (1,3)
            orthogonalize: bool - whether to orthogonalize blended R_i via SVD/polar (recommended)
        """
        self.body_pose = body_pose if body_pose is not None else self.body_pose
        self.global_orient = (
            global_orient if global_orient is not None else self.global_orient
        )
        self.transl = transl if transl is not None else self.transl

        device = self.device
        with torch.no_grad():
            out = self.model(
                betas=self.betas,
                body_pose=self.body_pose,
                global_orient=self.global_orient,
                transl=self.transl,
                return_verts=True,
                return_joints=True,
            )

        # posed vertex positions (global)
        self.vertices = out.vertices[0].detach().cpu().numpy()  # (V,3)
        self.joints = out.joints[0].detach()
        self.means, self.scales, self.quats = build_vertex_gaussians(
            self.vertices, self.faces, device=self.device
        )

    def normalize(self):
        center = (self.means.max(dim=0).values + self.means.min(dim=0).values) / 2
        ratio = 1 / torch.max(
            self.means.max(dim=0).values - self.means.min(dim=0).values + 1e-6
        )
        normalized_splats = {
            "means": (self.means - center) * ratio,
            "opacities": self.opacities,
            "scales": self.scales * ratio,
            "quats": self.quats,
            "joints": (self.joints - center) * ratio,
            "ratio": ratio,
            "center": center,
        }
        return normalized_splats

    def update_rest_attributes(self, colors):
        self.rest_colors = colors
        self.colors = colors


class SMPLinGaussianCube:

    def __init__(
        self,
        model_path,
        std_volume,
        gc_mean,
        gc_std,
        device,
        betas=None,
    ):
        self.smpl = SMPL(model_path, device=device, betas=betas)
        self.device = device
        self.splats = self.smpl.normalize()  # real splats
        self.std_volume = std_volume
        std_volume_offsets = gc_mean[:3, :, :, :].reshape(3, -1).T
        self.M = M = len(self.splats["means"])
        self.N = N = len(std_volume)
        self.voxel_size = int(np.round(N ** (1 / 3)))
        assert self.voxel_size**3 == N
        self.num_channels = len(gc_mean)
        self.assignments = recursive_unique_assignment(
            self.splats["means"], std_volume + std_volume_offsets, device=device
        )
        self.inversed_assignments = invert_assignments(
            self.assignments, len(std_volume)
        )
        matched_voxel_mask = self.inversed_assignments != -1
        matched_splat_indices = self.inversed_assignments[matched_voxel_mask]

        self.fixed_x0 = torch.ones((N, self.num_channels), device=device) * torch.nan
        self.fixed_x0[:, -8] = 0.0  # opacity set as 0
        self.fixed_x0[matched_voxel_mask] = torch.cat(
            [
                self.splats["means"][matched_splat_indices]
                - std_volume[matched_voxel_mask],  # fixed
                torch.ones(M, self.num_channels - 11, device=device)
                * torch.nan,  # to be generated
                self.splats["opacities"][matched_splat_indices],  # fixed
                self.splats["scales"][matched_splat_indices],  # fixed
                self.splats["quats"][matched_splat_indices],  # fixed
            ],
            dim=1,
        )
        # self.initial_x0 = torch.ones((N, self.num_channels), device=device) * torch.nan
        # self.initial_x0[matched_voxel_mask] = torch.cat(
        #     [
        #         torch.ones(M, 3, device=device) * torch.nan,  # already fixed
        #         torch.ones(M, self.num_channels - 11, device=device)
        #         * torch.nan,  # to be generated
        #         torch.ones(M, 1, device=device) * torch.nan,  # already fixed
        #         self.splats["scales"][matched_splat_indices],  # initial values
        #         self.splats["quats"][matched_splat_indices],  # initial values
        #     ],
        #     dim=1,
        # )
        # assert not torch.logical_and(
        #     ~self.fixed_x0.isnan(), ~self.initial_x0.isnan()
        # ).any()
        self.fixed_x0 = (
            self.fixed_x0 - gc_mean.view(self.num_channels, -1).T
        ) / gc_std.view(self.num_channels, -1).T
        self.fixed_x0 = (
            self.fixed_x0.view(
                self.voxel_size, self.voxel_size, self.voxel_size, self.num_channels
            )
            .permute(3, 0, 1, 2)
            .contiguous()
        )
        self.fixed_x0 = self.fixed_x0.unsqueeze(0)
        # self.initial_x0 = (
        #     self.initial_x0 - gc_mean.view(self.num_channels, -1).T
        # ) / gc_std.view(self.num_channels, -1).T
        # self.initial_x0 = (
        #     self.initial_x0.view(
        #         self.voxel_size, self.voxel_size, self.voxel_size, self.num_channels
        #     )
        #     .permute(3, 0, 1, 2)
        #     .contiguous()
        # )
        # self.initial_x0 = self.initial_x0.unsqueeze(0)

    def update_rest_attributes(self, x0_denorm, assignments=None):
        """
        x0_denorm: (num_channels, voxel_size, voxel_size, voxel_size)
        """
        x0_denorm = x0_denorm.permute(1, 2, 3, 0).reshape(-1, self.num_channels)
        assignments = self.assignments if assignments is None else assignments
        self.splats["colors"] = x0_denorm[assignments, 3 : self.num_channels - 8]
        self.smpl.update_rest_attributes(colors=self.splats["colors"])

    def apply_pose(self, body_pose=None, global_orient=None, transl=None):
        if "colors" not in self.splats:
            raise ValueError(
                "Rest attributes not updated. Call update_rest_attributes() first."
            )
        self.smpl.apply_pose(
            body_pose=body_pose, global_orient=global_orient, transl=transl
        )
        self.splats.update(self.smpl.normalize())

    def to_x0_denorm(self):
        x0_denorm = torch.zeros((self.N, self.num_channels), device=self.device)
        matched_voxel_mask = self.inversed_assignments != -1
        matched_splat_indices = self.inversed_assignments[matched_voxel_mask]
        x0_denorm[matched_voxel_mask] = torch.cat(
            [
                self.splats["means"][matched_splat_indices]
                - self.std_volume[matched_voxel_mask],
                self.splats["colors"][matched_splat_indices],
                self.splats["opacities"][matched_splat_indices],
                self.splats["scales"][matched_splat_indices],
                self.splats["quats"][matched_splat_indices],
            ],
            dim=1,
        )
        return x0_denorm.T.view(
            self.num_channels, self.voxel_size, self.voxel_size, self.voxel_size
        )
