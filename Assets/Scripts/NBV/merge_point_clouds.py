import sys
import open3d as o3d
import numpy as np

def voxelize_with_open3d(point_cloud, voxel_size=0.05):
    """
    Voxelizes a point cloud to remove duplicate/overlapping points.
    Args:
        point_cloud: open3d.geometry.PointCloud
        voxel_size: float, voxel size in same units as point cloud
    Returns:
        voxelized_cloud: open3d.geometry.PointCloud
    """
    return point_cloud.voxel_down_sample(voxel_size)

def get_point_cloud_scale(point_cloud):
    """
    Returns the axis-aligned bounding box size (scale) of the point cloud.
    """
    points = np.asarray(point_cloud.points)
    min_pt = points.min(axis=0)
    max_pt = points.max(axis=0)
    scale = max_pt - min_pt
    return scale, min_pt, max_pt

def main(cloud1_path, cloud2_path, output_path):
    # Load both point clouds
    pcd1 = o3d.io.read_point_cloud(cloud1_path)
    pcd2 = o3d.io.read_point_cloud(cloud2_path)

    # Merge points
    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)
    merged_points = np.vstack((points1, points2))

    # If colors exist, merge them; else, set all to white
    if pcd1.has_colors() and pcd2.has_colors():
        colors1 = np.asarray(pcd1.colors)
        colors2 = np.asarray(pcd2.colors)
        merged_colors = np.vstack((colors1, colors2))
    else:
        merged_colors = np.ones_like(merged_points)

    # Create merged point cloud
    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(merged_points)
    merged_pcd.colors = o3d.utility.Vector3dVector(merged_colors)

    # Print scale/size of merged point cloud
    scale, min_pt, max_pt = get_point_cloud_scale(merged_pcd)
    print(f"Merged point cloud bounding box:")
    print(f"  Min: {min_pt}")
    print(f"  Max: {max_pt}")
    print(f"  Scale (x, y, z): {scale}")
    print(f"  Largest dimension: {scale.max()}")

    # Voxelize to remove duplicates/overlaps
    avg_scale = scale.mean()
    voxel_size = 0.02 * avg_scale  # Adjust as needed
    voxelized_pcd = voxelize_with_open3d(merged_pcd, voxel_size)

    # Save voxelized point cloud
    o3d.io.write_point_cloud(output_path, voxelized_pcd)
    print(f"Merged & voxelized point cloud saved to {output_path}. Total points: {len(voxelized_pcd.points)}")

    # Note: If you have >2048 points, PoinTr will upsample/downsample to 2048 internally during inference. You do not need to manually downsample before running PoinTr.

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python merge_point_clouds.py <cloud1.ply> <cloud2.ply> <output.ply>")
        sys.exit(1)
    cloud1_path = sys.argv[1]
    cloud2_path = sys.argv[2]
    output_path = sys.argv[3]

    main(cloud1_path, cloud2_path, output_path)
