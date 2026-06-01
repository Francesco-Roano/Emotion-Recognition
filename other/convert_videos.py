import os
import subprocess

input_dir = "/home/roano/standalone/video_papa"
output_dir = "/home/roano/standalone/video_papa_converted"

os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    src = os.path.join(input_dir,file)
    dst = os.path.join(output_dir,file.replace(".MOV",".mp4"))
    if os.path.exists(dst):
        continue
    cmd = ["ffmpeg", "-y", "-i", src, "-vcodec", "libx264", "-acodec", "aac", dst]
    subprocess.run(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        # check=True solleva un'eccezione se il comando fallisce
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(f"Errore durante la conversione di {file}: {e}")

print("DONE CONVERTING")