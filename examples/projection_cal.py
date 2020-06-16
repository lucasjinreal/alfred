
import numpy as np



P = '7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00'
K = '9.842439e+02 0.000000e+00 6.900000e+02 0.000000e+00 9.808141e+02 2.331966e+02 0.000000e+00 0.000000e+00 1.000000e+00'
R = '9.999239e-01 9.837760e-03 -7.445048e-03 -9.869795e-03 9.999421e-01 -4.278459e-03 7.402527e-03 4.351614e-03 9.999631e-01'
T = '2.573699e-16 -1.059758e-16 1.614870e-16'

def get_m_from_str(s):
    ss = [np.float32(i) for i in s.split(' ')]
    ll = len(ss)
    ss = np.array(ss)
    ss = ss.reshape((3, ll//3))
    return ss


p = get_m_from_str(P)
print(p)

k = get_m_from_str(K)
r = get_m_from_str(R)
t = get_m_from_str(T)

r = np.vstack((r, [[0, 0, 0]]))
t = np.vstack((t, [[1]]))
rt = np.hstack([r, t])
k = np.hstack([k, [[0], [0], [0]]])

print(rt)
print(k)
c_p = np.dot(k, rt)
print(c_p)


