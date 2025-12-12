import os
from moviepy.editor import ImageClip, concatenate_videoclips

# Paths to frames you want to include in the demo video
# UPDATE THIS LIST with the exact .png files you want
frames = [
    "outputs/kitti_pointpillars_gpu/000008_2d_vis.png",
    "results/custom_pcd_lamppost_iso.png",
    "outputs/nuscenes_centerpoint/sample_2d_vis.png",
    "results/custom_pcd_roomscan1_iso.png"
]

clips = []
for f in frames:
    if os.path.exists(f):
        clips.append(ImageClip(f).set_duration(2))  # 2 sec per frame

if not clips:
    raise RuntimeError("No frames found; please check file paths.")

final = concatenate_videoclips(clips, method="compose")
final.write_videofile("results/demo_video.mp4", fps=24, codec="libx264", audio=False)

print("Demo video saved to results/demo_video.mp4")
