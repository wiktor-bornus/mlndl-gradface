
import torch
from PIL import Image

def extract_face_embedding(resnet, path, tolist=True, mtcnn=None):
    # try:
    img = Image.open(path).convert("RGB")
    face = mtcnn(img)
    face = face.cpu()
    print(face.shape)
    # face_to_plot  = (face + 1) / 2
    # plt.figure(figsize=(10, 10))
    # plt.subplot(221)
    # plt.imshow(img)
    #
    # plt.subplot(222)
    # plt.imshow(to_pil_image(face_to_plot))
    #
    # plt.subplot(223)
    # plt.hist(pil_to_tensor(img)[0, :, :].flatten(), bins=64, alpha=0.3, color="red")
    # plt.hist(pil_to_tensor(img)[1, :, :].flatten(), bins=64, alpha=0.3, color="green")
    # plt.hist(pil_to_tensor(img)[2, :, :].flatten(), bins=64, alpha=0.3, color="blue")
    # plt.grid(alpha=0.3)
    #
    # plt.subplot(224)
    # plt.hist(face_to_plot[0, :, :].flatten(), bins=64, alpha=0.3, color="red")
    # plt.hist(face_to_plot[1, :, :].flatten(), bins=64, alpha=0.3, color="green")
    # plt.hist(face_to_plot[2, :, :].flatten(), bins=64, alpha=0.3, color="blue")
    # plt.grid(alpha=0.3)



    if face is None:
        print(f"No face detected: {path}")
        return None

    face = face.unsqueeze(0).to('cuda')

    with torch.no_grad():
        embedding = resnet(face)

    # embedding = embedding / embedding.norm(dim=1, keepdim=True)

    if tolist:
        return embedding.squeeze().cpu().tolist()
    else:
        return embedding.squeeze()

    # except Exception as e:
    #     print(f"Error processing {path}: {e}")
    #     return None


import numpy as np


def extract_frontal_face(
    image,
    mtcnn,
    min_face_size_ratio=0.15,
    center_tolerance=0.35,
    max_roll_deg=15,
    max_yaw_ratio=0.5,
    pitch_range=(0.25, 0.75),
):
    """
    Parameters
    ----------
    image : PIL.Image
    mtcnn : facenet_pytorch.MTCNN

    Returns
    -------
    torch.Tensor | None
        Processed face tensor from mtcnn(image)
        or None if quality checks fail.
    """

    boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)

    if boxes is None:
        return None

    # require exactly one face
    if len(boxes) != 1:
        return None

    box = boxes[0]
    landmarks = landmarks[0]

    x1, y1, x2, y2 = box

    img_w, img_h = image.size

    # -------------------------
    # face size check
    # -------------------------

    face_w = x2 - x1
    face_h = y2 - y1

    if (
        face_w / img_w < min_face_size_ratio
        or face_h / img_h < min_face_size_ratio
    ):
        return None

    # -------------------------
    # center check
    # -------------------------

    face_cx = (x1 + x2) / 2
    face_cy = (y1 + y2) / 2

    img_cx = img_w / 2
    img_cy = img_h / 2

    dx = abs(face_cx - img_cx) / img_w
    dy = abs(face_cy - img_cy) / img_h

    if dx > center_tolerance or dy > center_tolerance:
        return None

    # -------------------------
    # landmarks
    # -------------------------

    left_eye = landmarks[0]
    right_eye = landmarks[1]
    nose = landmarks[2]
    mouth_left = landmarks[3]
    mouth_right = landmarks[4]

    # -------------------------
    # roll check
    # -------------------------

    eye_dx = right_eye[0] - left_eye[0]
    eye_dy = right_eye[1] - left_eye[1]

    roll_deg = np.degrees(np.arctan2(eye_dy, eye_dx))

    if abs(roll_deg) > max_roll_deg:
        return None

    # -------------------------
    # yaw check
    # -------------------------

    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    eye_distance = abs(right_eye[0] - left_eye[0])

    if eye_distance < 1:
        return None

    yaw_ratio = abs(nose[0] - eye_center_x) / eye_distance

    if yaw_ratio > max_yaw_ratio:
        return None

    # -------------------------
    # pitch check
    # -------------------------

    mouth_y = (mouth_left[1] + mouth_right[1]) / 2
    eye_y = (left_eye[1] + right_eye[1]) / 2

    denom = mouth_y - eye_y

    if denom <= 1:
        return None

    pitch_ratio = (nose[1] - eye_y) / denom

    if not (pitch_range[0] <= pitch_ratio <= pitch_range[1]):
        return None

    # -------------------------
    # final crop/alignment
    # -------------------------

    face_tensor = mtcnn(image)

    return face_tensor