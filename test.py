import os
import deeplake

os.environ["DEEPLAKE_DOWNLOAD_PATH"] = "/workspaces/SpeakerVerification/tmp"

ds = deeplake.load("hub://activeloop/coco-train", access_method="local")
