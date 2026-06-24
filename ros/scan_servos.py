#!/usr/bin/env python3
"""
Dynamixel Protocol 2.0 ping scan — 標準ライブラリのみ (termios/select)
外部パッケージ不要。demo.launch.py を先に止めること。
"""
import os, sys, struct, time, select, termios

DEVICE   = "/dev/sciurus17spine"
BAUD     = getattr(termios, "B3000000", 0x1006)   # 3 Mbps
WAIT_S   = 0.003   # 送信後待ち 3ms (エコー＋応答が届くまで)
READ_S   = 0.020   # 追加読み取り 20ms

SCAN_IDS = list(range(1, 21))
ID_NAME  = {
    2:  "r_arm_joint1  (XM540)",  3:  "r_arm_joint2  (XM540)",
    4:  "r_arm_joint3  (XM430)",  5:  "r_arm_joint4  (XM540)",
    6:  "r_arm_joint5  (XM430)",  7:  "r_arm_joint6  (XM430)",
    8:  "r_arm_joint7  (XM430)",  9:  "r_gripper     (XM430)",
    10: "l_arm_joint1  (XM540)", 11: "l_arm_joint2  (XM540)",
    12: "l_arm_joint3  (XM430)", 13: "l_arm_joint4  (XM540)",
    14: "l_arm_joint5  (XM430)", 15: "l_arm_joint6  (XM430)",
    16: "l_arm_joint7  (XM430)", 17: "l_gripper     (XM430)",
    18: "waist_yaw     (XM540)", 19: "neck_yaw      (XM430)",
    20: "neck_pitch    (XM430)",
}

# ── CRC-16 (Dynamixel Protocol 2.0) ──────────────────────────────────────────
_T = [
    0x0000,0x8005,0x800F,0x000A,0x801B,0x001E,0x0014,0x8011,
    0x8033,0x0036,0x003C,0x8039,0x0028,0x802D,0x8027,0x0022,
    0x8063,0x0066,0x006C,0x8069,0x0078,0x807D,0x8077,0x0072,
    0x0050,0x8055,0x805F,0x005A,0x804B,0x004E,0x0044,0x8041,
    0x80C3,0x00C6,0x00CC,0x80C9,0x00D8,0x80DD,0x80D7,0x00D2,
    0x00F0,0x80F5,0x80FF,0x00FA,0x80EB,0x00EE,0x00E4,0x80E1,
    0x00A0,0x80A5,0x80AF,0x00AA,0x80BB,0x00BE,0x00B4,0x80B1,
    0x8093,0x0096,0x009C,0x8099,0x0088,0x808D,0x8087,0x0082,
    0x8183,0x0186,0x018C,0x8189,0x0198,0x819D,0x8197,0x0192,
    0x01B0,0x81B5,0x81BF,0x01BA,0x81AB,0x01AE,0x01A4,0x81A1,
    0x01E0,0x81E5,0x81EF,0x01EA,0x81FB,0x01FE,0x01F4,0x81F1,
    0x81D3,0x01D6,0x01DC,0x81D9,0x01C8,0x81CD,0x81C7,0x01C2,
    0x0140,0x8145,0x814F,0x014A,0x815B,0x015E,0x0154,0x8151,
    0x8173,0x0176,0x017C,0x8179,0x0168,0x816D,0x8167,0x0162,
    0x8123,0x0126,0x012C,0x8129,0x0138,0x813D,0x8137,0x0132,
    0x0110,0x8115,0x811F,0x011A,0x810B,0x010E,0x0104,0x8101,
    0x8303,0x0306,0x030C,0x8309,0x0318,0x831D,0x8317,0x0312,
    0x0330,0x8335,0x833F,0x033A,0x832B,0x032E,0x0324,0x8321,
    0x0360,0x8365,0x836F,0x036A,0x837B,0x037E,0x0374,0x8371,
    0x8353,0x0356,0x035C,0x8359,0x0348,0x834D,0x8347,0x0342,
    0x03C0,0x83C5,0x83CF,0x03CA,0x83DB,0x03DE,0x03D4,0x83D1,
    0x83F3,0x03F6,0x03FC,0x83F9,0x03E8,0x83ED,0x83E7,0x03E2,
    0x83A3,0x03A6,0x03AC,0x83A9,0x03B8,0x83BD,0x83B7,0x03B2,
    0x0390,0x8395,0x839F,0x039A,0x838B,0x038E,0x0384,0x8381,
    0x0280,0x8285,0x828F,0x028A,0x829B,0x029E,0x0294,0x8291,
    0x82B3,0x02B6,0x02BC,0x82B9,0x02A8,0x82AD,0x82A7,0x02A2,
    0x02E0,0x82E5,0x82EF,0x02EA,0x82FB,0x02FE,0x02F4,0x82F1,
    0x82D3,0x02D6,0x02DC,0x82D9,0x02C8,0x82CD,0x82C7,0x02C2,
    0x0240,0x8245,0x824F,0x024A,0x825B,0x025E,0x0254,0x8251,
    0x8273,0x0276,0x027C,0x8279,0x0268,0x826D,0x8267,0x0262,
    0x8223,0x0226,0x022C,0x8229,0x0238,0x823D,0x8237,0x0232,
    0x0210,0x8215,0x821F,0x021A,0x820B,0x020E,0x0204,0x8201,
]
def _crc16(data: bytes) -> int:
    crc = 0
    for b in data:
        crc = ((crc << 8) ^ _T[((crc >> 8) ^ b) & 0xFF]) & 0xFFFF
    return crc

def _ping_pkt(dxl_id: int) -> bytes:
    body = bytes([0xFF, 0xFF, 0xFD, 0x00, dxl_id, 0x03, 0x00, 0x01])
    return body + struct.pack("<H", _crc16(body))

# ── シリアルポート ────────────────────────────────────────────────────────────
def open_port(device, baud_const):
    fd = os.open(device, os.O_RDWR | os.O_NOCTTY)
    a = termios.tcgetattr(fd)
    a[0] = 0
    a[1] = 0
    a[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
    a[3] = 0
    a[4] = a[5] = baud_const
    a[6][termios.VMIN] = 0
    a[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW, a)
    termios.tcflush(fd, termios.TCIOFLUSH)
    return fd

def _read_available(fd, total_wait):
    """total_wait 秒間、届いたバイトを全部読む"""
    buf = b""
    deadline = time.monotonic() + total_wait
    while True:
        rem = deadline - time.monotonic()
        if rem <= 0:
            break
        r, _, _ = select.select([fd], [], [], min(rem, 0.005))
        if r:
            buf += os.read(fd, 256)
    return buf

def ping(fd, dxl_id):
    termios.tcflush(fd, termios.TCIOFLUSH)
    os.write(fd, _ping_pkt(dxl_id))
    time.sleep(WAIT_S)                     # エコー＋応答到着を待つ
    raw = _read_available(fd, READ_S)      # 全バイト回収

    # ステータスパケットのヘッダを探す: FF FF FD 00 [id] 07 00 55
    needle = bytes([0xFF, 0xFF, 0xFD, 0x00, dxl_id, 0x07, 0x00, 0x55])
    pos = raw.find(needle)
    if pos >= 0 and len(raw) >= pos + 11:
        hw_err = raw[pos + 8]
        model  = struct.unpack_from("<H", raw, pos + 9)[0]
        return True, model, hw_err
    return False, 0, 0

# ── メイン ────────────────────────────────────────────────────────────────────
try:
    fd = open_port(DEVICE, BAUD)
except Exception as e:
    print(f"[エラー] {DEVICE} を開けません: {e}")
    sys.exit(1)

print(f"\n{'='*62}")
print(f"  Dynamixel スキャン  {DEVICE}  3 Mbps")
print(f"{'='*62}")
print(f"  {'ID':>3}  {'関節名':<26}  状態")
print(f"  {'-'*3}  {'-'*26}  {'-'*22}")

ok_ids, ng_ids = [], []
for dxl_id in SCAN_IDS:
    ok, model, hw_err = ping(fd, dxl_id)
    name = ID_NAME.get(dxl_id, "(未定義)")
    if ok:
        hw = f"  ← HW Error=0x{hw_err:02X}" if hw_err else ""
        print(f"  {dxl_id:>3}  {name:<26}  ✅ OK  model={model}{hw}")
        ok_ids.append(dxl_id)
    else:
        print(f"  {dxl_id:>3}  {name:<26}  ❌ 応答なし")
        ng_ids.append(dxl_id)

os.close(fd)
print(f"{'='*62}")
print(f"  応答あり: {ok_ids}")
print(f"  応答なし: {ng_ids}")
print(f"{'='*62}\n")
