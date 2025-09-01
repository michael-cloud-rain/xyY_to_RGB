
"""
xyY to RGB 色彩空間轉換程式
此程式將 xyY 色彩空間的值轉換為 RGB 色彩空間。
轉換流程：xyY -> XYZ -> Lab -> RGB

需求套件：
- PIL (Pillow)
- numpy
"""

# pip install pillow
from PIL import Image, ImageCms
import numpy as np
from pathlib import Path

# 常數定義
D50_REFERENCE_WHITE = {
    'X': 96.4212,
    'Y': 100.0,
    'Z': 82.5188
}

def validate_xyY(x, y, Y):
    if not (0 <= x <= 1):
        raise ValueError("x 值必須在 0 到 1 之間")
    if not (0 <= y <= 1):
        raise ValueError("y 值必須在 0 到 1 之間")
    if y == 0:
        raise ValueError("y 不能為 0")
    if Y < 0:
        raise ValueError("Y 必須大於或等於 0")

#xyY -> XYZ
try:
    xyY_input = input("請輸入xyY值，並以space鍵分隔 (x y Y): ")
    values = xyY_input.split()
    if len(values) != 3:
        raise ValueError("請輸入三個數值，以空格分隔")
    x, y, Y = map(float, values)
    validate_xyY(x, y, Y)
except ValueError as e:
    raise ValueError(f"輸入錯誤：{str(e)}")
else:
    try:
        X = round(x*Y/y, 2)
        Z = round((1 - x - y)/y*Y, 2)
    except ZeroDivisionError:
        raise ValueError("y 值不能為 0")
# ---- 0) 顯示器 ICC 路徑 ----
icc_path = r"C:\Windows\System32\spool\drivers\color\2091 #1 2025-08-30 23-00 2.2 F-S XYZLUT+MTX.icm"
assert Path(icc_path).exists(), f"ICC 不存在：{icc_path}"

# ---- 1) 你的目標：XYZ(D50) ----
XYZ_D50 = np.array([( X , Y , Z )], dtype="float32")  # 由 (x,y,Y) 轉來

# ---- 2) XYZ(D50) -> Lab(D50)（浮點；白點 D50）----
def xyz_to_lab_D50(xyz):  # xyz shape: (N,3)
    # D50 參考白（ICC/PCS）
    Xn = D50_REFERENCE_WHITE['X']
    Yn = D50_REFERENCE_WHITE['Y']
    Zn = D50_REFERENCE_WHITE['Z']
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
print("RGB(0-100%) = ",tuple(round(c*100/256, 4) for c in rgb255))
