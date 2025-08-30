# # pip install pillow
# from PIL import ImageCms
# import numpy as np
# from pathlib import Path
# import platform

# # ---------- 0) 你的 ICC 路徑 ----------
# icc_path = r"C:\Windows\System32\spool\drivers\color\CS2420 #1 2025-08-22 16-07 2.2 F-S XYZLUT+MTX.icm"  # ←改成你的檔案

# # 32-bit Python 讀 System32 會被轉向，做個防呆
# if platform.system() == "Windows" and platform.architecture()[0] == "32bit" and icc_path.startswith(r"C:\Windows\System32"):
#     icc_path = icc_path.replace(r"C:\Windows\System32", r"C:\Windows\Sysnative")
# assert Path(icc_path).exists(), f"ICC 不存在：{icc_path}"

# # ---------- 1) 你的目標：XYZ(D50) ----------
# XYZ_D50 = np.array([(26.4015, 27.7481, 13.4673)], dtype="float32")  # 由 (x,y,Y) 轉來

# # ---------- 2) 準備 profiles 與 intent ----------
# dst = ImageCms.getOpenProfile(icc_path)
# src_xyz = ImageCms.createProfile("XYZ")    # D50 XYZ
# src_lab = ImageCms.createProfile("LAB")    # D50 Lab

# INTENT_ABS = getattr(ImageCms, "INTENT_ABSOLUTE_COLORIMETRIC", 3)
# INTENT_REL = getattr(ImageCms, "INTENT_RELATIVE_COLORIMETRIC", 1)

# # ---------- 3) 先試：XYZ -> RGB（Absolute，再來 Relative） ----------
# xform = None
# mode_in = None
# last_err = None

# try:
#     xform = ImageCms.buildTransformFromOpenProfiles(src_xyz, dst, "XYZ", "RGB",
#                                                     renderingIntent=INTENT_ABS)
#     mode_in = "XYZ"
# except Exception as e:
#     last_err = e

# if xform is None:
#     try:
#         xform = ImageCms.buildTransformFromOpenProfiles(src_xyz, dst, "XYZ", "RGB",
#                                                         renderingIntent=INTENT_REL)
#         mode_in = "XYZ"
#     except Exception as e:
#         last_err = e

# # ---------- 4) 都不行：手算 XYZ(D50)→Lab(D50)，再 Lab -> RGB（Relative） ----------
# def xyz_to_lab_D50(xyz):  # xyz: (N,3) float32
#     # 參考白：D50（ICC/PCS）
#     Xn, Yn, Zn = 96.4212, 100.0, 82.5188
#     x = xyz[:, 0] / Xn
#     y = xyz[:, 1] / Yn
#     z = xyz[:, 2] / Zn
#     e = (6/29)**3          # 0.008856...
#     k = 24389/27           # 903.3...
#     def f(t):
#         return np.where(t > e, np.cbrt(t), (k*t + 16) / 116)
#     fx, fy, fz = f(x), f(y), f(z)
#     L = 116*fy - 16
#     a = 500*(fx - fy)
#     b = 200*(fy - fz)
#     lab = np.stack([L, a, b], axis=1).astype("float32")
#     return lab

# if xform is None:
#     try:
#         xform = ImageCms.buildTransformFromOpenProfiles(src_lab, dst, "LAB", "RGB",
#                                                         renderingIntent=INTENT_REL)
#         mode_in = "LAB"
#     except Exception as e:
#         last_err = e

# if xform is None:
#     # 還是失敗就把最後一次錯誤拋出，便於定位
#     raise last_err

# # ---------- 5) 套用轉換，得到像素 RGB ----------
# if mode_in == "XYZ":
#     src_data = XYZ_D50
# else:  # "LAB"
#     src_data = xyz_to_lab_D50(XYZ_D50)

# rgb01 = ImageCms.applyTransform(src_data, xform)         # shape: (1,3)
# rgb255 = np.clip(np.round(rgb01 * 255), 0, 255).astype(int).ravel()
# print("RGB(0–255) =", tuple(rgb255))

# pip install pillow
from PIL import Image, ImageCms
import numpy as np
from pathlib import Path

# ---- 0) 顯示器 ICC 路徑 ----
icc_path = r"C:\Windows\System32\spool\drivers\color\CS2420 #1 2025-08-28 17-08 0.3127x 0.329y sRGB F-S XYZLUT+MTX.icm"
assert Path(icc_path).exists(), f"ICC 不存在：{icc_path}"

# ---- 1) 你的目標：XYZ(D50) ----
XYZ_D50 = np.array([(7.16, 7.68, 2.62)], dtype="float32")  # 由 (x,y,Y) 轉來

# ---- 2) XYZ(D50) -> Lab(D50)（浮點；白點 D50）----
def xyz_to_lab_D50(xyz):  # xyz shape: (N,3)
    # D50 參考白（ICC/PCS）
    Xn, Yn, Zn = 96.4212, 100.0, 82.5188
    x = xyz[:, 0] / Xn
    y = xyz[:, 1] / Yn
    z = xyz[:, 2] / Zn
    e = (6/29)**3
    k = 24389/27
    f = lambda t: np.where(t > e, np.cbrt(t), (k*t + 16) / 116)
    fx, fy, fz = f(x), f(y), f(z)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    return np.stack([L, a, b], axis=1).astype("float32")

LAB_f = xyz_to_lab_D50(XYZ_D50)  # -> [[L, a, b]]

# ---- 3) 把浮點 Lab 轉成 Pillow 的 8-bit 'LAB' 影像（一個像素）----
L8  = int(np.clip(round(LAB_f[0,0] * 255/100), 0, 255))
a8  = int(np.clip(round(LAB_f[0,1] + 128),      0, 255))
b8  = int(np.clip(round(LAB_f[0,2] + 128),      0, 255))
lab_img = Image.new("LAB", (1,1))
lab_img.putpixel((0,0), (L8, a8, b8))

# ---- 4) 建 transform：LAB -> RGB（Relative intent）----
src_lab = ImageCms.createProfile("LAB")    # D50 Lab
dst     = ImageCms.getOpenProfile(icc_path)
INTENT_REL = getattr(ImageCms, "INTENT_RELATIVE_COLORIMETRIC", 1)

xform = ImageCms.buildTransformFromOpenProfiles(
    src_lab, dst, "LAB", "RGB", renderingIntent=INTENT_REL
)

# ---- 5) 套用變換，得到像素 RGB(0–255) ----
rgb_img = ImageCms.applyTransform(lab_img, xform)   # 回傳 1×1 RGB 影像
rgb255  = rgb_img.getpixel((0,0))                   # 取得 (R,G,B)
print("RGB(0–255) =", rgb255)
