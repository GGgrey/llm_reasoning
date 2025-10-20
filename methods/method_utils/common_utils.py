import random


def get_random_points(length, num):
    pts = []
    mu = (length - 1) / 2
    sigma = length / 6
    while True:
        idx = int(random.gauss(mu, sigma))
        if 0 <= idx < length:
            if length >= num:  # Make each point different
                if idx not in pts:
                    pts.append(idx)
            else:
                pts.append(idx)
        if len(pts) == num:
            break
    
    return pts