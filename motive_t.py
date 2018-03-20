"""LCM type definitions
This file automatically generated by lcm.
DO NOT MODIFY BY HAND!!!!
"""

try:
    import cStringIO.StringIO as BytesIO
except ImportError:
    from io import BytesIO
import struct

class motive_t(object):
    __slots__ = ["quat_x", "quat_y", "quat_z", "quat_w", "x_pos", "y_pos", "z_pos", "euler_alpha", "euler_beta", "euler_gamma"]

    def __init__(self):
        self.quat_x = 0.0
        self.quat_y = 0.0
        self.quat_z = 0.0
        self.quat_w = 0.0
        self.x_pos = 0.0
        self.y_pos = 0.0
        self.z_pos = 0.0
        self.euler_alpha = 0.0
        self.euler_beta = 0.0
        self.euler_gamma = 0.0

    def encode(self):
        buf = BytesIO()
        buf.write(motive_t._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf):
        buf.write(struct.pack(">dddddddddd", self.quat_x, self.quat_y, self.quat_z, self.quat_w, self.x_pos, self.y_pos, self.z_pos, self.euler_alpha, self.euler_beta, self.euler_gamma))

    def decode(data):
        if hasattr(data, 'read'):
            buf = data
        else:
            buf = BytesIO(data)
        if buf.read(8) != motive_t._get_packed_fingerprint():
            raise ValueError("Decode error")
        return motive_t._decode_one(buf)
    decode = staticmethod(decode)

    def _decode_one(buf):
        self = motive_t()
        self.quat_x, self.quat_y, self.quat_z, self.quat_w, self.x_pos, self.y_pos, self.z_pos, self.euler_alpha, self.euler_beta, self.euler_gamma = struct.unpack(">dddddddddd", buf.read(80))
        return self
    _decode_one = staticmethod(_decode_one)

    _hash = None
    def _get_hash_recursive(parents):
        if motive_t in parents: return 0
        tmphash = (0x51235ea0cb1db7ce) & 0xffffffffffffffff
        tmphash  = (((tmphash<<1)&0xffffffffffffffff)  + (tmphash>>63)) & 0xffffffffffffffff
        return tmphash
    _get_hash_recursive = staticmethod(_get_hash_recursive)
    _packed_fingerprint = None

    def _get_packed_fingerprint():
        if motive_t._packed_fingerprint is None:
            motive_t._packed_fingerprint = struct.pack(">Q", motive_t._get_hash_recursive([]))
        return motive_t._packed_fingerprint
    _get_packed_fingerprint = staticmethod(_get_packed_fingerprint)
