from lirics.euclid import Cell, Frame

# TODO : Refactor, maybe after development of Bernoulli?

COEFF_A = 1.4


def compute_hydraulic_diameter(length, width):
    return 4*width*length/(length+2*width)


def compute_k_roughness(roughness, hydraulic_diameter, Reynolds):

    k = (1*(Reynolds <= 4e4) +
         (1+roughness/hydraulic_diameter*2e6)*(Reynolds > 4e4))

    return k


def compute_k_Reynolds(Reynolds):

    k = 20.3/Reynolds**0.25*(Reynolds <= 2e5)+1*(Reynolds > 2e5)

    return k


def compute_coeff_B(rim_radius, hydraulic_diameter):

    non_dim_realtion = rim_radius/hydraulic_diameter
    n = -2.5*(non_dim_realtion <= 1) - 0.5*(non_dim_realtion > 1)
    B = 0.21*non_dim_realtion**n

    return B


def compute_coeff_C(length, average_width):

    non_dim_relation = length/(2*average_width)
    C = (
        0.223*(non_dim_relation)**0.448*(non_dim_relation >= 3) +
        non_dim_relation*(non_dim_relation < 3))

    return C


def compute_friction_coeff(roughness, hydraulic_diameter, Reynolds):

    coeff = 0.1*(1.46*roughness/hydraulic_diameter+100/Reynolds)**0.25

    return coeff


def compute_friction_loss(roughness, hydraulic_diameter, Reynolds,
                          frame_radius, rim_radius, angular_step):

    friction_coeff = compute_friction_coeff(
        roughness, hydraulic_diameter, Reynolds)

    loss = (friction_coeff*1/2*(frame_radius+rim_radius) /
            hydraulic_diameter*angular_step)

    return loss


def compute_local_loss(roughness, hydraulic_diameter, Reynolds,
                       rim_radius, length, average_width, angular_step):

    k_roughness = compute_k_roughness(
        roughness, hydraulic_diameter, Reynolds)
    k_Reynolfs = compute_k_Reynolds(Reynolds)

    A = COEFF_A
    B = compute_coeff_B(rim_radius, hydraulic_diameter)
    C = compute_coeff_C(length, average_width)

    loss = k_roughness*k_Reynolfs*A*B*C*angular_step

    return loss
