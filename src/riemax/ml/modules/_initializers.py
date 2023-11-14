from haiku.initializers import VarianceScaling

glorot_uniform = VarianceScaling(1.0, 'fan_avg', 'uniform')
glorot_normal = VarianceScaling(1.0, 'fan_avg', 'truncated_normal')
lecun_uniform = VarianceScaling(1.0, 'fan_in', 'uniform')
lecun_normal = VarianceScaling(1.0, 'fan_in', 'truncated_normal')
he_uniform = VarianceScaling(2.0, 'fan_in', 'uniform')
he_normal = VarianceScaling(2.0, 'fan_in', 'truncated_normal')
