import math
import numpy as np
from datetime import datetime, timedelta
import time

start_time = time.time()

# Input parameters
input_date = datetime(2025, 6, 12, 8, 24, 30)
latitude = 23.035904
longitude = 72.542908
timezone_offset = 5.5  # IST = UTC+5.5
elevation = 55.1  # Observer's elevation in meters
temp_c = 35.5  # Temperature in Celsius
p_hpa = 1008   # Pressure in hPa
omega_deg = 23  # Surface tilt angle
gamma_deg = 180 
delta_T = 69

DEG_TO_RAD = math.pi / 180
RAD_TO_DEG = 180 / math.pi

# === Time Conversion Functions ===
def lst_to_utc(lst_dt, tz_offset):
    return lst_dt - timedelta(hours=tz_offset)

# === Julian Date Calculations ===
def julian_date(dt):
    year = dt.year
    month = dt.month
    day = dt.day + dt.hour / 24 + dt.minute / 1440 + dt.second / 86400
    
    if month <= 2:
        year -= 1
        month += 12
        
    A = int(year / 100)
    B = 2 - A + int(A / 4)
    jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5
    return jd

# === Time System Conversions ===
def julian_century(jd):
    return (jd - 2451545.0) / 36525.0

def julian_ephemeris_day(jd, delta_T):
    return jd + delta_T / 86400.0

def julian_ephemeris_century(jde):
    return (jde - 2451545.0) / 36525.0

def julian_ephemeris_millennium(jce):
    return jce / 10.0

# === Earth Heliocentric Coordinates ===
L_TERMS = {
    0: [(175347046, 0, 0), (3341656, 4.6692568, 6283.07585), (34894, 4.6261, 12566.1517), 
        (3497, 2.7441, 5753.3849), (3418, 2.8289, 3.5231), (3136, 3.6277, 77713.7715),
        (2676, 4.4181, 7860.4194), (2343, 6.1352, 3930.2097), (1324, 0.7425, 11506.7698),
        (1273, 2.0371, 529.691), (1199, 1.1096, 1577.3435), (990, 5.233, 5884.927),
        (902, 2.045, 26.298), (857, 3.508, 398.149), (780, 1.179, 5223.694),
        (753, 2.533, 5507.553), (505, 4.583, 18849.228), (492, 4.205, 775.523),
        (357, 2.92, 0.067), (317, 5.849, 11790.629), (284, 1.899, 796.298),
        (271, 0.315, 10977.079), (243, 0.345, 5486.778), (206, 4.806, 2544.314),
        (205, 1.869, 5573.143), (202, 2.458, 6069.777), (156, 0.833, 213.299),
        (132, 3.411, 2942.463), (126, 1.083, 20.775), (115, 0.645, 0.98),
        (103, 0.636, 4694.003), (102, 0.976, 15720.839), (102, 4.267, 7.114),
        (99, 6.21, 2146.17), (98, 0.68, 155.42), (86, 5.98, 161000.69),
        (85, 1.3, 6275.96), (85, 3.67, 71430.7), (80, 1.81, 17260.15),
        (79, 3.04, 12036.46), (75, 1.76, 5088.63), (74, 3.5, 3154.69),
        (74, 4.68, 801.82), (70, 0.83, 9437.76), (62, 3.98, 8827.39),
        (61, 1.82, 7084.9), (57, 2.78, 6286.6), (56, 4.39, 14143.5),
        (56, 3.47, 6279.55), (52, 0.19, 12139.55), (52, 1.33, 1748.02),
        (51, 0.28, 5856.48), (49, 0.49, 1194.45), (41, 5.37, 8429.24),
        (41, 2.4, 19651.05), (39, 6.17, 10447.39), (37, 6.04, 10213.29),
        (37, 2.57, 1059.38), (36, 1.71, 2352.87), (36, 1.78, 6812.77),
        (33, 0.59, 17789.85), (30, 0.44, 83996.85), (30, 2.74, 1349.87),
        (25, 3.16, 4690.48)],
    1: [(628331966747, 0, 0), (206059, 2.678235, 6283.07585), (4303, 2.6351, 12566.1517),
        (425, 1.59, 3.523), (119, 5.796, 26.298), (109, 2.966, 1577.344),
        (93, 2.59, 18849.23), (72, 1.14, 529.69), (68, 1.87, 398.15),
        (67, 4.41, 5507.55), (59, 2.89, 5223.69), (56, 2.17, 155.42),
        (45, 0.4, 796.3), (36, 0.47, 775.52), (29, 2.65, 7.11),
        (21, 5.34, 0.98), (19, 1.85, 5486.78), (19, 4.97, 213.3),
        (17, 2.99, 6275.96), (16, 0.03, 2544.31), (16, 1.43, 2146.17),
        (15, 1.21, 10977.08), (12, 2.83, 1748.02), (12, 3.26, 5088.63),
        (12, 5.27, 1194.45), (12, 2.08, 4694), (11, 0.77, 553.57),
        (10, 1.3, 6286.6), (10, 4.24, 1349.87), (9, 2.7, 242.73),
        (9, 5.64, 951.72), (8, 5.3, 2352.87), (6, 2.65, 9437.76),
        (6, 4.67, 4690.48)],
    2: [(52919, 0, 0), (8720, 1.0721, 6283.0758), (309, 0.867, 12566.152),
        (27, 0.05, 3.52), (16, 5.19, 26.3), (16, 3.68, 155.42),
        (10, 0.76, 18849.23), (9, 2.06, 77713.77), (7, 0.83, 775.52),
        (5, 4.66, 1577.34), (4, 1.03, 7.11), (4, 3.44, 5573.14),
        (3, 5.14, 796.3), (3, 6.05, 5507.55), (3, 1.19, 242.73),
        (3, 6.12, 529.69), (3, 0.31, 398.15), (3, 2.28, 553.57),
        (2, 4.38, 5223.69), (2, 3.75, 0.98)],
    3: [(289, 5.844, 6283.076), (35, 0, 0), (17, 5.49, 12566.15),
        (3, 5.2, 155.42), (1, 4.72, 3.52), (1, 5.3, 18849.23),
        (1, 5.97, 242.73)],
    4: [(114, 3.142, 0), (8, 4.13, 6283.08), (1, 3.84, 12566.15)],
    5: [(1.0, 3.14, 0)]
}

B_TERMS = {
    0: [(280, 3.199, 84334.662), (102, 5.422, 5507.553), (80, 3.88, 5223.69),
        (44, 3.7, 2352.87), (32, 4, 1577.34)],
    1: [(9.0, 3.9, 5507.55), (6, 1.73, 5223.69)],
    2: [], 3: [], 4: [], 5: []
}

R_TERMS = {
    0: [(100013989, 0, 0), (1670700, 3.0984635, 6283.07585), (13956, 3.05525, 12566.1517),
        (3084, 5.1985, 77713.7715), (1628, 1.1739, 5753.3849), (1576, 2.8469, 7860.4194),
        (925, 5.453, 11506.77), (542, 4.564, 3930.21), (472, 3.661, 5884.927),
        (346, 0.964, 5507.553), (329, 5.9, 5223.694), (307, 0.299, 5573.143),
        (243, 4.273, 11790.629), (212, 5.847, 1577.344), (186, 5.022, 10977.079),
        (175, 3.012, 18849.228), (110, 5.055, 5486.778), (98, 0.89, 6069.78),
        (86, 5.69, 15720.84), (86, 1.27, 161000.69), (65, 0.27, 17260.15),
        (63, 0.92, 529.69), (57, 2.01, 83996.85), (56, 5.24, 71430.7),
        (49, 3.25, 2544.31), (47, 2.58, 775.52), (45, 5.54, 9437.76),
        (43, 6.01, 6275.96), (39, 5.36, 4694), (38, 2.39, 8827.39),
        (37, 0.83, 19651.05), (37, 4.9, 12139.55), (36, 1.67, 12036.46),
        (35, 1.84, 2942.46), (33, 0.24, 7084.9), (32, 0.18, 5088.63),
        (32, 1.78, 398.15), (28, 1.21, 6286.6), (28, 1.9, 6279.55),
        (26, 4.59, 10447.39)],
    1: [(103019, 1.10749, 6283.07585), (1721, 1.0644, 12566.1517), (702, 3.142, 0),
        (32, 1.02, 18849.23), (31, 2.84, 5507.55), (25, 1.32, 5223.69),
        (18, 1.42, 1577.34), (10, 5.91, 10977.08), (9, 1.42, 6275.96),
        (9, 0.27, 5486.78)],
    2: [(4359, 5.7846, 6283.0758), (124, 5.579, 12566.152), (12, 3.14, 0),
        (9, 3.63, 77713.77), (6, 1.87, 5573.14), (3, 5.47, 18849.23)],
    3: [(145, 4.273, 6283.076), (7, 3.92, 12566.15)],
    4: [(4.0, 2.56, 6283.08)], 5: []
}

def compute_series(terms, JME):
    total = 0.0
    for n in range(6):
        term_sum = 0.0
        for A, B, C in terms.get(n, []):
            term_sum += A * math.cos(B + C * JME)
        total += term_sum * (JME ** n)
    return total

def compute_LBR(JME):
    L = compute_series(L_TERMS, JME) / 1e8
    B = compute_series(B_TERMS, JME) / 1e8
    R = compute_series(R_TERMS, JME) / 1e8
    
    L_deg = math.degrees(L) % 360
    B_deg = math.degrees(B)
    
    theta = L_deg + 180  # Geocentric longitude (λ)
    beta = -B_deg        # Geocentric latitude (β)
    
    return R, theta, beta, L_deg, B_deg

# === Nutation Calculations ===
nutation_coeffs = [
    [0, 0, 0, 0, 1, -171996, -174.2, 92025, 8.9], [-2, 0, 0, 2, 2, -13187, -1.6, 5736, -3.1],
    [0, 0, 0, 2, 2, -2274, -0.2, 977, -0.5], [0, 0, 0, 0, 2, 2062, 0.2, -895, 0.5],
    [0, 1, 0, 0, 0, 1426, -3.4, 54, -0.1], [0, 0, 1, 0, 0, 712, 0.1, -7, 0],
    [-2, 1, 0, 2, 2, -517, 1.2, 224, -0.6], [0, 0, 0, 2, 1, -386, -0.4, 200, 0],
    [0, 0, 1, 2, 2, -301, 0, 129, -0.1], [-2, -1, 0, 2, 2, 217, -0.5, -95, 0.3],
    [-2, 0, 1, 0, 0, -158, 0, 0, 0], [-2, 0, 0, 2, 1, 129, 0.1, -70, 0],
    [0, 0, -1, 2, 2, 123, 0, -53, 0], [2, 0, 0, 0, 0, 63, 0, 0, 0],
    [0, 0, 1, 0, 1, 63, 0.1, -33, 0], [2, 0, -1, 2, 2, -59, 0, 26, 0],
    [0, 0, -1, 0, 1, -58, -0.1, 32, 0], [0, 0, 1, 2, 1, -51, 0, 27, 0],
    [-2, 0, 2, 0, 0, 48, 0, 0, 0], [0, 0, -2, 2, 1, 46, 0, -24, 0],
    [2, 0, 0, 2, 2, -38, 0, 16, 0], [0, 0, 2, 2, 2, -31, 0, 13, 0],
    [0, 0, 2, 0, 0, 29, 0, 0, 0], [-2, 0, 1, 2, 2, 29, 0, -12, 0],
    [0, 0, 0, 2, 0, 26, 0, 0, 0], [-2, 0, 0, 2, 0, -22, 0, 0, 0],
    [0, 0, -1, 2, 1, 21, 0, -10, 0], [0, 2, 0, 0, 0, 17, -0.1, 0, 0],
    [2, 0, -1, 0, 1, 16, 0, -8, 0], [-2, 2, 0, 2, 2, -16, 0.1, 7, 0],
    [0, 1, 0, 0, 1, -15, 0, 9, 0], [-2, 0, 1, 0, 1, -13, 0, 7, 0],
    [0, -1, 0, 0, 1, -12, 0, 6, 0], [0, 0, 2, -2, 0, 11, 0, 0, 0],
    [2, 0, -1, 2, 1, -10, 0, 5, 0], [2, 0, 1, 2, 2, -8, 0, 3, 0],
    [0, 1, 0, 2, 2, 7, 0, -3, 0], [-2, 1, 1, 0, 0, -7, 0, 0, 0],
    [0, -1, 0, 2, 2, -7, 0, 3, 0], [2, 0, 0, 2, 1, -7, 0, 3, 0],
    [2, 0, 1, 0, 0, 6, 0, 0, 0], [-2, 0, 2, 2, 2, 6, 0, -3, 0],
    [-2, 0, 1, 2, 1, 6, 0, -3, 0], [2, 0, -2, 0, 1, -6, 0, 3, 0],
    [2, 0, 0, 0, 1, -6, 0, 3, 0], [0, -1, 1, 0, 0, 5, 0, 0, 0],
    [-2, -1, 0, 2, 1, -5, 0, 3, 0], [-2, 0, 0, 0, 1, -5, 0, 3, 0],
    [0, 0, 2, 2, 1, -5, 0, 3, 0], [-2, 0, 2, 0, 1, 4, 0, 0, 0],
    [-2, 1, 0, 2, 1, 4, 0, 0, 0], [0, 0, 1, -2, 0, 4, 0, 0, 0],
    [-1, 0, 1, 0, 0, -4, 0, 0, 0], [-2, 1, 0, 0, 0, -4, 0, 0, 0],
    [1, 0, 0, 0, 0, -4, 0, 0, 0], [0, 0, 1, 2, 0, 3, 0, 0, 0],
    [0, 0, -2, 2, 2, -3, 0, 0, 0], [-1, -1, 1, 0, 0, -3, 0, 0, 0],
    [0, 1, 1, 0, 0, -3, 0, 0, 0], [0, -1, 1, 2, 2, -3, 0, 0, 0],
    [2, -1, -1, 2, 2, -3, 0, 0, 0], [0, 0, 3, 2, 2, -3, 0, 0, 0],
    [2, -1, 0, 2, 2, -3, 0, 0, 0]
]

def compute_nutation(JCE):
    X0 = 297.85036 + 445267.111480 * JCE - 0.0019142 * JCE**2 + JCE**3 / 189474
    X1 = 357.52772 + 35999.050340 * JCE - 0.0001603 * JCE**2 - JCE**3 / 300000
    X2 = 134.96298 + 477198.867398 * JCE + 0.0086972 * JCE**2 + JCE**3 / 56250
    X3 = 93.27191 + 483202.017538 * JCE - 0.0036825 * JCE**2 + JCE**3 / 327270
    X4 = 125.04452 - 1934.136261 * JCE + 0.0020708 * JCE**2 + JCE**3 / 450000
    
    X_deg = [X0, X1, X2, X3, X4]
    X_rad = [math.radians(x) for x in X_deg]
    
    delta_psi_i = []
    delta_epsilon_i = []
    
    for row in nutation_coeffs:
        Y = np.array(row[:5])
        a, b, c, d = row[5:]
        arg = np.dot(X_rad, Y)
        delta_psi_i.append((a + b * JCE) * math.sin(arg))
        delta_epsilon_i.append((c + d * JCE) * math.cos(arg))
    
    delta_psi = sum(delta_psi_i) / 36000000  # in degrees
    delta_epsilon = sum(delta_epsilon_i) / 36000000  # in degrees
    
    return delta_psi, delta_epsilon

def mean_obliquity_ecliptic(JME):
    U = JME / 10.0
    epsilon_not = (84381.448 - U * 4680.93 - 1.55 * U**2 + 1999.25 * U**3 - 51.38 * U**4 - 
                  249.67 * U**5 - 39.05 * U**6 + 7.12 * U**7 + 27.87 * U**8 + 5.79 * U**9 + 2.45 * U**10)
    return epsilon_not  # In arcseconds

def true_obliquity(epsilon_not_arcsec, delta_epsilon_deg):
    return (epsilon_not_arcsec / 3600) + delta_epsilon_deg

def delta_tau(R):
    return -20.4898 / (3600 * R)

def apparent_sun_longitude(theta, delta_psi, delta_tau):
    return theta + delta_psi + delta_tau

def mean_sidereal_time(JD, JC):
    ν0 = (280.46061837 + 360.98564736629 * (JD - 2451545) + 0.000387933 * JC**2 - JC**3 / 38710000)
    return ν0 % 360

def apparent_sidereal_time(ν0, delta_psi, true_eps_deg):
    return ν0 + delta_psi * math.cos(math.radians(true_eps_deg))

def sun_right_ascension(apparent_sun_longitude, true_eps_deg, beta_deg):
    lambda_rad = math.radians(apparent_sun_longitude)
    eps_rad = math.radians(true_eps_deg)
    beta_rad = math.radians(beta_deg)
    
    alpha_rad = math.atan2(
        math.sin(lambda_rad) * math.cos(eps_rad) - math.tan(beta_rad) * math.sin(eps_rad),
        math.cos(lambda_rad)
    )
    return math.degrees(alpha_rad) % 360

def sun_declination(apparent_sun_longitude, true_eps_deg, beta_deg):
    lambda_rad = math.radians(apparent_sun_longitude)
    eps_rad = math.radians(true_eps_deg)
    beta_rad = math.radians(beta_deg)
    
    decl_rad = math.asin(
        math.sin(beta_rad) * math.cos(eps_rad) + 
        math.cos(beta_rad) * math.sin(eps_rad) * math.sin(lambda_rad)
    )
    return math.degrees(decl_rad)

def hour_angle(ν, longitude, alpha):
    H = ν + longitude - alpha
    return H % 360

def parallax_in_altitude(R):
    return 8.794 / (3600 * R)

def auxiliary_angle_u(latitude):
    return math.atan(0.99664719 * math.tan(math.radians(latitude)))

def term_x(u, latitude, elevation):
    return math.cos(u) + (elevation / 6378140.0) * math.cos(math.radians(latitude))

def term_y(u, latitude, elevation):
    return 0.99664719 * math.sin(u) + (elevation / 6378140.0) * math.sin(math.radians(latitude))

def delta_alpha(xi, parallax, H, decl_rad):
    return math.degrees(math.atan2(
        -xi * math.sin(math.radians(parallax)) * math.sin(math.radians(H)),
        math.cos(decl_rad) - xi * math.sin(math.radians(parallax)) * math.cos(math.radians(H))
    ))

def alpha_prime(alpha, delta_alpha):
    return alpha + delta_alpha

def decl_prime(decl_rad, eta, parallax, delta_alpha_rad, xi, H):
    return math.atan2(
        math.sin(decl_rad) - eta * math.sin(math.radians(parallax)) * math.cos(delta_alpha_rad),
        math.cos(decl_rad) - xi * math.sin(math.radians(parallax)) * math.cos(math.radians(H))
    )

def hour_angle_prime(H, delta_alpha):
    return H - delta_alpha

def topocentric_elevation(latitude, decl_prime, H_prime):
    return math.degrees(math.asin(
        math.sin(math.radians(latitude)) * math.sin(decl_prime) + 
        math.cos(math.radians(latitude)) * math.cos(decl_prime) * math.cos(math.radians(H_prime))
    ))

def atmospheric_refraction_correction(e0, pressure_hPa, temp_C):
    if e0 > 0:
        del_e = (pressure_hPa / 1010.0) * (283.0 / (273.0 + temp_C)) * (
            1.02 / (60 * math.tan(math.radians(e0 + 10.3 / (e0 + 5.11)))))
    else:
        del_e = 0.0
    return del_e

def topocentric_zenith_angle(e):
    return 90 - e

def topocentric_azimuth_angle(latitude, H_prime, decl_prime):
    azimuth = math.degrees(math.atan2(
        math.sin(math.radians(H_prime)),
        math.cos(math.radians(H_prime)) * math.sin(math.radians(latitude)) - 
        math.tan(decl_prime) * math.cos(math.radians(latitude))
    )) % 360
    return (azimuth + 180) % 360  # Convert to measured from south

def incidence_angle(theta, omega, gamma, azimuth):
    return math.degrees(math.acos(
        math.cos(math.radians(theta)) * math.cos(math.radians(omega)) + 
        math.sin(math.radians(omega)) * math.sin(math.radians(theta)) * 
        math.cos(math.radians(azimuth - gamma))
    ))

def equation_of_time(JME, alpha, delta_psi, true_eps):
    E = 280.4664567 + 360007.6982779 * JME + 0.03032028 * JME**2 + JME**3/49931 - JME**4/15300 - JME**5/2000000
    return (E - 0.0057183 - alpha + delta_psi * math.cos(math.radians(true_eps))) % 360

# === Main Calculation ===
ut_time = lst_to_utc(input_date, timezone_offset)
JD = julian_date(ut_time)
JC = julian_century(JD)
JDE = julian_ephemeris_day(JD, delta_T)
JCE = julian_ephemeris_century(JDE)
JME = julian_ephemeris_millennium(JCE)

# Earth heliocentric position
R, theta, beta, L_deg, B_deg = compute_LBR(JME)

# Nutation
delta_psi, delta_epsilon = compute_nutation(JCE)

# Obliquity
epsilon_not = mean_obliquity_ecliptic(JME)
true_eps = true_obliquity(epsilon_not, delta_epsilon)

# Apparent sun longitude
delta_tau_val = delta_tau(R)
lambda_apparent = apparent_sun_longitude(theta, delta_psi, delta_tau_val)

# Sidereal time
nu0 = mean_sidereal_time(JD, JC)
nu = apparent_sidereal_time(nu0, delta_psi, true_eps)

# Sun right ascension and declination
alpha = sun_right_ascension(lambda_apparent, true_eps, beta)
decl = sun_declination(lambda_apparent, true_eps, beta)
decl_rad = math.radians(decl)

# Hour angle
H = hour_angle(nu, longitude, alpha)

# Topocentric calculations
parallax = parallax_in_altitude(R)
u = auxiliary_angle_u(latitude)
xi = term_x(u, latitude, elevation)
eta = term_y(u, latitude, elevation)
delta_alpha_val = delta_alpha(xi, parallax, H, decl_rad)
alpha_prime_val = alpha_prime(alpha, delta_alpha_val)
H_prime = hour_angle_prime(H, delta_alpha_val)
decl_prime_val = decl_prime(decl_rad, eta, parallax, math.radians(delta_alpha_val), xi, H)

# Topocentric elevation (before refraction)
e0 = topocentric_elevation(latitude, decl_prime_val, H_prime)

# Atmospheric refraction
del_e = atmospheric_refraction_correction(e0, p_hpa, temp_c)

# Final topocentric elevation
e = e0 + del_e

# Other angles
theta_zenith = topocentric_zenith_angle(e)
azimuth = topocentric_azimuth_angle(latitude, H_prime, decl_prime_val)
incidence = incidence_angle(theta_zenith, omega_deg, gamma_deg, azimuth)

# Equation of time
E = equation_of_time(JME, alpha, delta_psi, true_eps)

def calculate_solar_noon_sunrise_sunset(latitude, longitude, timezone_offset, date, delta_T, elevation=0, temp_c=15, p_hpa=1013.25):
    # Convert input date to UTC at midnight (00:00:00)
    local_midnight = datetime(date.year, date.month, date.day, 0, 0, 0)
    utc_midnight = lst_to_utc(local_midnight, timezone_offset)
    
    # Calculate Julian Date for midnight UTC
    jd_midnight = julian_date(utc_midnight)
    
    # Approximate solar noon (in fraction of day)
    # First estimate - solar noon is approximately at local noon (12:00) minus equation of time
    # We'll do an iterative approach to find the exact time
    
    # Function to calculate transit (solar noon) time
    def calculate_transit_time(jd_start):
        # Iterate to find exact transit time
        prev_ha = 360  # Initialize with large value
        delta = 0
        for _ in range(10):  # 10 iterations should be sufficient
            jd = jd_start + delta
            jc = julian_century(jd)
            jde = julian_ephemeris_day(jd, delta_T)
            jce = julian_ephemeris_century(jde)
            jme = julian_ephemeris_millennium(jce)
            
            # Calculate sun position
            R, theta, beta, L_deg, B_deg = compute_LBR(jme)
            delta_psi, delta_epsilon = compute_nutation(jce)
            epsilon_not = mean_obliquity_ecliptic(jme)
            true_eps = true_obliquity(epsilon_not, delta_epsilon)
            delta_tau_val = delta_tau(R)
            lambda_apparent = apparent_sun_longitude(theta, delta_psi, delta_tau_val)
            alpha = sun_right_ascension(lambda_apparent, true_eps, beta)
            
            # Calculate sidereal time
            nu0 = mean_sidereal_time(jd, jc)
            nu = apparent_sidereal_time(nu0, delta_psi, true_eps)
            
            # Hour angle
            H = (nu + longitude - alpha) % 360
            if H > 180:
                H -= 360
            
            # Update delta
            delta += -H / 360.0
            if abs(H) < 0.0001:  # Sufficient precision
                break
            prev_ha = H
        
        return jd_start + delta
    
    # Calculate transit time (solar noon)
    transit_jd = calculate_transit_time(jd_midnight + 0.5)  # Start with local noon estimate
    transit_utc = (transit_jd - jd_midnight) * 24  # Hours since midnight UTC
    
    # Function to calculate sunrise/sunset time
    def calculate_rise_set_time(jd_start, is_sunrise=True):
        # Iterate to find exact rise/set time
        prev_alt = 0
        delta = 0
        for _ in range(10):  # 10 iterations should be sufficient
            jd = jd_start + delta
            jc = julian_century(jd)
            jde = julian_ephemeris_day(jd, delta_T)
            jce = julian_ephemeris_century(jde)
            jme = julian_ephemeris_millennium(jce)
            
            # Calculate sun position
            R, theta, beta, L_deg, B_deg = compute_LBR(jme)
            delta_psi, delta_epsilon = compute_nutation(jce)
            epsilon_not = mean_obliquity_ecliptic(jme)
            true_eps = true_obliquity(epsilon_not, delta_epsilon)
            delta_tau_val = delta_tau(R)
            lambda_apparent = apparent_sun_longitude(theta, delta_psi, delta_tau_val)
            alpha = sun_right_ascension(lambda_apparent, true_eps, beta)
            decl = sun_declination(lambda_apparent, true_eps, beta)
            
            # Calculate sidereal time
            nu0 = mean_sidereal_time(jd, jc)
            nu = apparent_sidereal_time(nu0, delta_psi, true_eps)
            
            # Hour angle when sun is at horizon (including refraction)
            cos_ha = (math.sin(math.radians(-0.8333 - (2.076 * math.sqrt(elevation))/60)) - 
                     math.sin(math.radians(latitude)) * math.sin(math.radians(decl))) / (
                     math.cos(math.radians(latitude)) * math.cos(math.radians(decl)))
            
            if cos_ha < -1:  # Sun doesn't rise
                return None
            if cos_ha > 1:  # Sun doesn't set
                return None
                
            ha = math.degrees(math.acos(cos_ha))
            if is_sunrise:
                ha = -ha
            
            # Calculate time difference from transit
            time_diff = ha / 360.0  # Fraction of day
            
            # Update delta
            delta = time_diff
            if abs(ha - prev_alt) < 0.0001:  # Sufficient precision
                break
            prev_alt = ha
        
        return jd_start + 0.5 + delta  # 0.5 is transit time
    
    # Calculate sunrise and sunset times
    sunrise_jd = calculate_rise_set_time(jd_midnight, is_sunrise=True)
    sunset_jd = calculate_rise_set_time(jd_midnight, is_sunrise=False)
    
    # Convert JD times to datetime objects
    def jd_to_datetime(jd):
        if jd is None:
            return None
        jd -= jd_midnight
        hours = jd * 24
        return local_midnight + timedelta(hours=hours)
    
    solar_noon = jd_to_datetime(transit_jd)
    sunrise = jd_to_datetime(sunrise_jd) if sunrise_jd else None
    sunset = jd_to_datetime(sunset_jd) if sunset_jd else None
    
    return {
        'solar_noon': solar_noon,
        'sunrise': sunrise,
        'sunset': sunset
    }

# Add this to your main code
sun_times = calculate_solar_noon_sunrise_sunset(
    latitude=latitude,
    longitude=longitude,
    timezone_offset=timezone_offset,
    date=input_date,
    delta_T=delta_T,
    elevation=elevation,
    temp_c=temp_c,
    p_hpa=p_hpa
)

# Print results
print("\nAdditional Solar Times:")
print(f"Solar Noon: {sun_times['solar_noon'].strftime('%Y-%m-%d %H:%M:%S')}")
if sun_times['sunrise']:
    print(f"Sunrise: {sun_times['sunrise'].strftime('%Y-%m-%d %H:%M:%S')}")
else:
    print("Sunrise: Sun does not rise on this day")
if sun_times['sunset']:
    print(f"Sunset: {sun_times['sunset'].strftime('%Y-%m-%d %H:%M:%S')}")
else:
    print("Sunset: Sun does not set on this day")
print(f"Topocentric Elevation Angle: {e:.10f} degrees")
print(f"Topocentric Zenith Angle: {theta_zenith:.10f} degrees")
print(f"Topocentric Azimuth Angle: {azimuth:.10f} degrees (measured westward from south)")
print(f"Incidence Angle: {incidence:.10f} degrees")
print(f"Equation of Time: {E:.10f} minutes")
for i in range(1000000):
    pass  # Example task

end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time:.10f} seconds")