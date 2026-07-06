"""Core steganography engine.

Pipeline (Algorithm 1 of the project objectives):
  secret (image or text) -> bytes -> binary -> 2's complement -> XOR with
  pixel MSBs -> written into the low bits of cover-image pixels selected by
  the ML region model.

Extraction reverses every step. All operations are exactly invertible, so
recovery of the secret is bit-perfect (100% extraction accuracy) as long as
the stego image is stored losslessly (PNG).
"""

import numpy as np
import cv2

MAGIC = b"\xab\x57"          # packet marker: identifies a valid stego image
TYPE_TEXT = 0
TYPE_IMAGE = 1
HEADER_BYTES = 8             # MAGIC(2) + TYPE(1) + BPC(1) + LENGTH(4)
HEADER_SLOTS = HEADER_BYTES * 8   # header is always 1 bit/slot in raster order


def twos_complement(data: np.ndarray) -> np.ndarray:
    """Apply the 2's complement transform to every byte: S2 = ~S + 1.

    The transform is its own inverse modulo 256, so the same function
    decodes what it encoded.
    """
    return ((256 - data.astype(np.int32)) % 256).astype(np.uint8)


def bytes_to_bits(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    arr = twos_complement(arr)
    return np.unpackbits(arr)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    arr = np.packbits(bits)
    arr = twos_complement(arr)
    return arr.tobytes()


def build_packet(payload: bytes, ptype: int, bpc: int) -> bytes:
    header = MAGIC + bytes([ptype, bpc]) + len(payload).to_bytes(4, "big")
    return header + payload


def capacity_bytes(img: np.ndarray, bpc: int = 1) -> int:
    """Payload capacity of a cover image at bpc bits per channel-slot."""
    slots = img.size - HEADER_SLOTS
    return max(0, (slots * bpc) // 8)


def _write_bits(flat: np.ndarray, positions: np.ndarray, bits: np.ndarray, bpc: int):
    """Write bits into the low `bpc` bits of flat[positions], each bit XORed
    with the MSB of the original pixel value (the MSB is untouched by the
    write, so the decoder can recompute it)."""
    n = len(positions)
    msb = (flat[positions] >> 7) & 1
    groups = bits.reshape(n, bpc)
    vals = np.zeros(n, dtype=np.uint8)
    for k in range(bpc):
        vals = (vals << 1) | (groups[:, k] ^ msb)
    mask = np.uint8((0xFF << bpc) & 0xFF)
    flat[positions] = (flat[positions] & mask) | vals


def _read_bits(flat: np.ndarray, positions: np.ndarray, bpc: int) -> np.ndarray:
    msb = (flat[positions] >> 7) & 1
    out = []
    for k in range(bpc - 1, -1, -1):
        out.append(((flat[positions] >> k) & 1) ^ msb)
    return np.stack(out, axis=1).reshape(-1)


def embed(cover: np.ndarray, payload: bytes, ptype: int, slot_order: np.ndarray,
          bpc: int = 1) -> np.ndarray:
    """Embed payload bytes into a cover image.

    slot_order: permutation of payload slot indices (from the ML region
    model), excluding the reserved header slots.
    """
    if bpc not in (1, 2):
        raise ValueError("bpc must be 1 or 2")
    cap = capacity_bytes(cover, bpc)
    if len(payload) > cap:
        raise ValueError(
            f"Payload is {len(payload)} bytes but this cover can hide at most "
            f"{cap} bytes at {bpc} bit(s) per channel.")

    stego = cover.copy()
    flat = stego.reshape(-1)

    header = MAGIC + bytes([ptype, bpc]) + len(payload).to_bytes(4, "big")
    header_bits = bytes_to_bits(header)
    _write_bits(flat, np.arange(HEADER_SLOTS), header_bits, 1)

    payload_bits = bytes_to_bits(payload)
    n_slots = int(np.ceil(len(payload_bits) / bpc))
    pad = n_slots * bpc - len(payload_bits)
    if pad:
        payload_bits = np.concatenate([payload_bits, np.zeros(pad, dtype=np.uint8)])
    _write_bits(flat, slot_order[:n_slots], payload_bits, bpc)
    return stego


def extract_header(stego: np.ndarray):
    """Read the fixed-position header. Returns (ptype, bpc, length) or None
    if the magic marker is absent (no hidden data)."""
    flat = stego.reshape(-1)
    bits = _read_bits(flat, np.arange(HEADER_SLOTS), 1)
    raw = bits_to_bytes(bits)
    if raw[:2] != MAGIC:
        return None
    ptype, bpc = raw[2], raw[3]
    length = int.from_bytes(raw[4:8], "big")
    if ptype not in (TYPE_TEXT, TYPE_IMAGE) or bpc not in (1, 2):
        return None
    return ptype, bpc, length


def extract(stego: np.ndarray, slot_order: np.ndarray):
    """Recover (ptype, payload bytes) from a stego image."""
    header = extract_header(stego)
    if header is None:
        raise ValueError("No hidden data found in this image (magic marker missing).")
    ptype, bpc, length = header
    cap = capacity_bytes(stego, bpc)
    if length > cap:
        raise ValueError("Corrupt header: declared payload exceeds image capacity.")
    n_bits = length * 8
    n_slots = int(np.ceil(n_bits / bpc))
    bits = _read_bits(stego.reshape(-1), slot_order[:n_slots], bpc)[:n_bits]
    return ptype, bits_to_bytes(bits)


# ---------------------------------------------------------------- payload io

def image_to_payload(secret_bgr: np.ndarray, cover: np.ndarray, bpc: int) -> bytes:
    """Encode a secret image as lossless PNG bytes, downscaling if necessary
    until it fits the cover's capacity. Recovery is bit-perfect for the
    (possibly resized) secret."""
    cap = capacity_bytes(cover, bpc)
    img = secret_bgr
    for _ in range(24):
        ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise ValueError("Could not PNG-encode the secret image.")
        if buf.size <= cap:
            return buf.tobytes()
        h, w = img.shape[:2]
        scale = max(0.5, (cap / buf.size) ** 0.5 * 0.95)
        nh, nw = max(8, int(h * scale)), max(8, int(w * scale))
        if (nh, nw) == (h, w):
            nh, nw = h - 1, w - 1
        if nh < 8 or nw < 8:
            break
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    raise ValueError(
        f"Secret image cannot fit: cover capacity is only {cap} bytes at "
        f"{bpc} bit(s)/channel. Use a larger cover or higher bits-per-channel.")


def payload_to_image(payload: bytes) -> np.ndarray:
    arr = np.frombuffer(payload, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Recovered payload is not a valid image.")
    return img
