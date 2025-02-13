"""Functionality related to the DHI EUM scientific type system.


Examples
--------
>>> mikeio.EUMType.Temperature
Temperature
>>> mikeio.EUMType.Temperature.value
100006
>>> mikeio.EUMType.Temperature.units
[degree Celsius, degree Fahrenheit, degree Kelvin]
>>> mikeio.EUMUnit.degree_Celsius
degree Celsius
>>> mikeio.EUMUnit.degree_Celsius.value
2800
>>>

"""

from __future__ import annotations
from enum import IntEnum
from typing import Any, Sequence, Literal

import pandas as pd
from mikecore.DfsFile import DataValueType, DfsDynamicItemInfo
from mikecore.eum import eumUnit, eumItem, eumWrapper

from ..exceptions import InvalidDataValueType


def _type_list(search: str | None = None) -> dict[eumItem, str]:
    """Get a dictionary of the EUM items.

    Notes
    -----
    An alternative to `type_list` is to use `mikeio.eum.Item`

    Parameters
    ----------
    search: str, optional
        a search string (caseinsensitive) to filter out results

    Returns
    -------
    dict
        names and codes for EUM items

    """
    items = {}
    check = True
    i = 1
    while check:
        d = eumWrapper.eumGetItemTypeSeq(i)
        if d[0] is True:
            items[d[1]] = d[2]
            i += 1
        else:
            check = False

    if search is not None:
        search = search.lower()
        items = dict(
            [
                [key, value]
                for key, value in items.items()
                if search in value.lower() or search == value.lower()
            ]
        )

    return items


def _unit_list(eum_type: int) -> dict[str, eumUnit]:
    """Get a dictionary of valid units.

    Parameters
    ----------
    eum_type: int
        EUM variable type, e.g. 100006 or EUMType.Temperature

    Returns
    -------
    dict
        names and codes for valid units

    """
    items = {}
    n_units_for_eum_type = eumWrapper.eumGetItemUnitCount(eum_type)
    for i in range(n_units_for_eum_type):
        _, value, key = eumWrapper.eumGetItemUnitSeq(eum_type, i + 1)
        items[key] = value

    return items


class TimeAxisType(IntEnum):
    EquidistantRelative = 1
    NonEquidistantRelative = 2
    EquidistantCalendar = 3
    NonEquidistantCalendar = 4

    def __repr__(self) -> str:
        return self.name


class TimeStepUnit(IntEnum):
    SECOND = 1400
    MINUTE = 1401
    HOUR = 1402
    DAY = 1403
    MONTH = 1405
    YEAR = 1404


class EUMType(IntEnum):
    """EUM type.

    Examples
    --------
    ```{python}
    import mikeio
    mikeio.EUMType.Temperature
    ```

    ```{python}
    mikeio.EUMType.Temperature.units
    ```

    """

    Water_Level = 100000
    Discharge = 100001
    Wind_Velocity = 100002
    Wind_Direction = 100003
    Rainfall = 100004
    Evaporation = 100005
    Temperature = 100006
    Concentration = 100007
    Bacteria_Concentration = 100008
    Resistance_factor = 100009
    Sediment_Transport = 100010
    Bottom_level = 100011
    Bottom_level_change = 100012
    Sediment_fraction = 100013
    Sediment_fraction_change = 100014
    Gate_Level = 100015
    Flow_velocity = 100016
    Density = 100017
    Dam_breach_level = 100018
    Dam_breach_width = 100019
    Dam_breach_slope = 100020
    Sunshine = 100021
    Sun_radiation = 100022
    Relative_humidity = 100023
    Salinity = 100024
    Surface_Slope = 100025
    Flow_Area = 100026
    Flow_Width = 100027
    Hydraulic_Radius = 100028
    Resistance_Radius = 100029
    Mannings_M = 100030
    Mannings_n = 100031
    Chezy_No = 100032
    Conveyance = 100033
    Froude_No = 100034
    Water_Volume = 100035
    Flooded_Area = 100036
    Water_Volume_Error = 100037
    Acc_Water_Volume_Error = 100038
    Component_Mass = 100039
    Component_Mass_Error = 100040
    Acc_Component_Mass_Error = 100041
    Relative_Component_Mass_Error = 100042
    Relative_Acc_Component_Mass_Error = 100043
    Component_Decay = 100044
    Acc_Component_Decay = 100045
    Component_Transport = 100046
    Acc_Component_Transport = 100047
    Component_Disp_Transport = 100048
    Acc_Component_Disp_Transport = 100049
    Component_Conv_Transport = 100050
    Acc_Component_Conv_Transport = 100051
    Acc_Sediment_transport = 100052
    Dune_length = 100053
    Dune_height = 100054
    Bed_sediment_load = 100055
    Suspended_sediment_load = 100056
    Irrigation = 100057
    Relative_moisture_content = 100058
    Ground_water_depth = 100059
    Snow_Water_Content = 100060
    Infiltration = 100061
    Recharge = 100062
    OF1_Flow = 100063
    IF1_Flow = 100064
    CapillaryFlux = 100065
    SurfStorage_OF1 = 100066
    SurfStorage_OF0 = 100067
    Sediment_layer = 100068
    Bed_level = 100069
    Rainfall_Intensity = 100070
    Production_rate = 100071
    Sediment_mass = 100072
    Primary_production = 100073
    Vol_specific_prod_rate = 100074
    Secchi_depth = 100075
    Acc_Sediment_Mass = 100076
    Sediment_Mass_per_m = 100077
    Surface_Elevation = 100078
    Bathymetry = 100079
    Flow_Flux = 100080
    Bed_sediment_load_per_m = 100081
    Suspended_load_per_m = 100082
    Sediment_transport_per_m = 100083
    Wave_height = 100084
    Wave_period = 100085
    Wave_frequency = 100086
    Potential_evaporation_rate = 100087
    Rainfall_rate = 100088
    Water_Flow = 100089
    Return_Flow_Fraction = 100090
    Linear_Routing_Coefficient = 100091
    Specific_runoff = 100092
    Machine_Efficiency = 100093
    Target_power = 100094
    Wave_direction = 100095
    Accumulated_transport_per_meter = 100096
    Significant_wave_height = 100097
    Critical_Shields_parameter = 100098
    AngleBedVelocity = 100099
    Profile_number = 100100
    Climate_number = 100101
    Spectral_description = 100102
    Spreading_factor = 100103
    Reference_point_number = 100104
    Wind_friction_factor = 100105
    Wave_Disturbance_Coefficient = 100106
    Time_of_first_wave_arrival = 100107
    Surface_Curvature = 100108
    Radiation_Stress = 100109
    Spectral_density = 100120
    Frequency_integrated_spectral_density = 100121
    Directional_integrated_spectral_density = 100122
    Viscosity = 100123
    Standard_deviation_DSD = 100124
    Beach_position = 100125
    Trench_position = 100126
    Grain_diameter = 100127
    Settling_velocity = 100128
    Geometrical_deviation = 100129
    Breaking_wave = 100130
    Dune_position = 100131
    Contour_angle = 100132
    Flow_direction = 100133
    Bed_slope = 100134
    Surface_area = 100135
    Catchment_area = 100136
    Roughness = 100137
    Active_Depth = 100138
    Sediment_Gradation = 100139
    Groundwater_recharge = 100140
    Solute_flux = 100141
    River_structure_geometry = 100142
    River_chainage = 100143
    Dimensionless_factor = 100144
    Dimensionless_exponent = 100145
    Storage_depth = 100146
    River_width = 100147
    Flow_routing_time_constant = 100148
    _1st_order_rate_AD_model = 100149
    _1st_order_rate_WQ_model = 100150
    Erosion_deposition_coefficient = 100151
    Shear_stress = 100152
    Dispersion_coefficient = 100153
    Dispersion_factor = 100154
    Sediment_volume_per_length_unit = 100155
    Latitude_longitude = 100157
    Specific_gravity = 100158
    Transmission_coefficient = 100159
    Reflection_coefficient = 100160
    Friction_factor = 100161
    Radiation_intensity = 100162
    Duration = 100163
    Respiration_production_per_area = 100164
    Respiration_production_per_volume = 100165
    Sediment_depth = 100166
    Angle_of_repose = 100167
    Half_order_rate_WQ_model = 100168
    Rearation_constant = 100169
    Deposition_rate = 100170
    BOD_at_river_bed = 100171
    Crop_demand = 100172
    Irrigated_area = 100173
    Livestock_demand = 100174
    Number_of_livestock = 100175
    Total_Gas = 100176
    Ground_water_abstraction = 100177
    Melting_coefficient = 100178
    Rain_melting_coefficient_per_degree_per_time = 100179
    Elevation = 100180
    Cross_section_X_data = 100181
    Vegetation_height = 100182
    Geographical_coordinate = 100183
    Angle = 100184
    ItemGeometry0D = 100185
    ItemGeometry1D = 100186
    ItemGeometry2D = 100187
    ItemGeometry3D = 100188
    Temperature_lapse_rate = 100189
    Correction_of_precipitation = 100190
    Temperature_correction = 100191
    Precipitation_correction = 100192
    Max_Water = 100193
    Lower_Baseflow = 100194
    Mass_flux = 100195
    Pressure2 = 100196
    Turbulent_kinetic_energy = 100197
    Dissipation_of_TKE = 100198
    Salt_Flux = 100199
    Temperature_Flux = 100200
    Concentration_Non_Dim = 100201
    Latent_Heat = 100202
    Heat_Flux = 100203
    Specific_Heat = 100204
    Visibility = 100205
    Ice_thickness = 100206
    Structure_geometry___time = 100207
    Discharge___time = 100208
    Fetch_length = 100209
    Rubble_mound = 100210
    Grid_Spacing = 100211
    TimeStep = 100212
    Length_Scale = 100213
    Erosion_Coefficient_Factor = 100214
    Friction_Coefficient = 100215
    Transition_Rate = 100216
    Distance = 100217
    Time_Correction_At_Noon = 100218
    Critical_Velocity = 100219
    Light_Extinction_Background = 100220
    Particle_Production_Rate = 100221
    First_Order_Grazing_Rate_Dependance = 100222
    Resuspension_Rate = 100223
    Adsorption_Coefficient = 100224
    Desorption_Coefficient = 100225
    Sedimentation_Velocity = 100226
    Boundary_Layer_Thickness = 100227
    Diffusion_Coefficient = 100228
    Bioconcentration_Factor = 100229
    Fcoli_Concentration = 100230
    Specific_Discharge = 100231
    Precipitation = 100232
    Specific_Precipitation = 100233
    Power = 100234
    Conveyance_Loss = 100235
    Infiltration_Flux = 100236
    Evaporation_Flux = 100237
    Ground_Water_Abstraction_Flux = 100238
    Fraction = 100239
    Yield_Factor = 100240
    Specific_Solute_Flux_per_Area = 100241
    Current_Speed = 100242
    Current_Direction = 100243
    Current_Magnitude = 100244
    First_Order_Piston_Position = 100245
    Subharmonic_Piston_Position = 100246
    Superharmonic_Piston_Position = 100247
    First_Order_Flap_Position = 100248
    Subharmonic_Flap_Position = 100249
    Superharmonic_Flap_Position = 100250
    Length_Zero_Crossing = 100251
    Time_Zero_Crossing = 100252
    Length_Logged_Data = 100253
    Force_Logged_Data = 100254
    Speed_Logged_Data = 100255
    Volume_Flow_Logged_Data = 100256
    _2D_Surface_Elevation_Spectrum = 100257
    _3D_Surface_Elevation_Spectrum = 100258
    Directional_Spreading_Function = 100259
    Auto_Spectrum = 100260
    Cross_Spectrum = 100261
    Coherence_Spectrum = 100262
    Coherent_Spectrum = 100263
    Frequency_Response_Spectrum = 100264
    Phase_Spectrum = 100265
    FIR_Filter_coefficient = 100266
    Fourier_a_Coefficient = 100267
    Fourier_b_Coefficient = 100268
    u_velocity_component = 100269
    v_velocity_component = 100270
    w_velocity_component = 100271
    Bed_Thickness = 100272
    Dispersion_Velocity_Factor = 100273
    Wind_speed = 100274
    Shore_Current_Zone = 100275
    Depth_of_Wind = 100276
    Emulsification_Constant_K1 = 100277
    Emulsification_Constant_K2 = 100278
    Light_Extinction = 100279
    Water_Depth = 100280
    Reference_settling_velocity = 100281
    Phase_Error = 100282
    Level_Amplitude_Error = 100283
    Discharge_Amplitude_Error = 100284
    Level_Correction = 100285
    Discharge_Correction = 100286
    Level_Simulated = 100287
    Discharge_Simulated = 100288
    Summ_Q_Corrected = 100289
    Time_Scale = 100290
    Sponge_Coefficient = 100291
    Porosity_Coefficient = 100292
    Filter_Coefficient = 100293
    Skewness = 100294
    Asymmetry = 100295
    Atiltness = 100296
    Kurtosis = 100297
    Auxiliary_variable_w = 100298
    Roller_Thickness = 100299
    Line_Thickness = 100300
    Marker_Size = 100301
    Roller_Celerity = 100302
    Encroachment_offset = 100303
    Encroachment_position = 100304
    Encroachment_width = 100305
    Conveyance_reduction = 100306
    Water_level_change = 100307
    Energy_level_change = 100308
    Horizontal_particle_velocity = 100309
    Vertical_particle_velocity = 100310
    Area_fraction = 100311
    Catchment_slope = 100312
    Average_length = 100313
    PE = 100314
    Inverse_exponent = 100315
    Time_shift = 100316
    Attenuation = 100317
    Population = 100318
    Industrial_output = 100319
    Agricultural_area = 100320
    Population_usage = 100321
    Industrial_use = 100322
    Agricultural_usage = 100323
    Layer_Thickness = 100324
    Snow_Depth = 100325
    Snow_Cover_Percentage = 100326
    Pressure_Head = 100353
    Crop_Coefficient = 100354
    Aroot_Kristensen_and_Jensen = 100355
    C1_Kristensen_and_Jensen = 100356
    C2_Kristensen_and_Jensen = 100357
    C3_Kristensen_and_Jensen = 100358
    Irrigation_Demand = 100359
    Transmissivity = 100360
    Darcy_Velocity = 100361
    Leakage_Coeff_Drain_Time_Const = 100362
    Conductance = 100363
    Height_Above_Ground = 100364
    Pumping_Rate = 100365
    Depth_Below_Ground = 100366
    Cell_Height = 100367
    Head_Gradient = 100368
    Ground_Water_Flow_Velocity = 100369
    Grid_Codes = 100370
    Drainage_Time_Constant = 100371
    Head_Elevation = 100372
    Length_Error = 100373
    Storage_Coefficient = 100374
    Specific_Yield = 100375
    Exchange_Rate = 100376
    Volumetric_Water_Content = 100377
    Storage_Change_Rate = 100378
    Seepage = 100379
    Root_Depth = 100380
    Rill_Depth = 100381
    Logical = 100382
    Leaf_Area_Index = 100383
    Irrigation_Rate = 100384
    Irrigation_Index = 100385
    Interception = 100386
    Evapotranspiration_Rate = 100387
    Erosion_Surface_Load = 100388
    Erosion_Concentration = 100389
    Epsilon_UZ = 100390
    Drainage = 100391
    Deficit = 100392
    Crop_Yield = 100393
    Crop_Type = 100394
    Crop_Stress = 100395
    Crop_Stage = 100396
    Crop_Loss = 100397
    Crop_Index = 100398
    Age = 100399
    Conductivity = 100400
    Print_Scale_Equivalence = 100401
    Concentration_1 = 100402
    Concentration_2 = 100403
    Concentration_3 = 100404
    Concentration_4 = 100405
    Sediment_diameter = 100406
    Mean_Wave_Direction = 100407
    Flow_Direction_1 = 100408
    Air_Pressure = 100409
    Decay_Factor = 100410
    Sediment_Bed_Density = 100411
    Dispersion_Coefficient = 100412
    Velocity_Profile = 100413
    Habitat_Index = 100414
    Angles = 100415
    Hydraulic_Length = 100416
    SCS_Catchment_slope = 100417
    Turbidity_FTU = 100418
    Turbidity_MgPerL = 100419
    Bacteria_Flow = 100420
    Bed_Distribution = 100421
    Surface_Elevation_at_Paddle = 100422
    Unit_Hydrograph_Response = 100423
    Transfer_Rate = 100424
    Return_period = 100425
    Constant_Settling_Velocity = 100426
    Deposition_Concentration_Flux = 100427
    Settling_Velocity_Coefficient = 100428
    Erosion_Coefficient = 100429
    Volume_Flux = 100430
    Precipitation_Rate = 100431
    Evaporation_Rate = 100432
    Co_Spectrum = 100433
    Quad_Spectrum = 100434
    Propagation_Direction = 100435
    Directional_Spreading = 100436
    Mass_per_Unit_Area = 100437
    Incident_Spectrum = 100438
    Reflected_Spectrum = 100439
    Reflection_Function = 100440
    Bacteria_Flux = 100441
    Head_Difference = 100442
    Energy = 100443
    Directional_Standard_Deviation = 100444
    Rainfall_Depth = 100445
    Ground_Water_Abstraction_Depth = 100446
    Evapo_Transpiration = 100447
    Longitudinal_Infiltration = 100448
    Pollutant_Load = 100449
    Pressure = 100450
    Cost_per_Time = 100451
    Mass = 100452
    Mass_per_Time = 100453
    Mass_per_Area_per_Time = 100454
    Kd = 100455
    Porosity = 100456
    Half_Life = 100457
    Dispersivity = 100458
    Friction_Coefficient_cfw = 100459
    Wave_amplitude = 100460
    Sediment_grain_diameter = 100461
    Sediment_spill = 100463
    Number_of_Particles = 100464
    Ellipsoidal_height = 100500
    Cloudiness = 100501
    Probability = 100502
    Activity_of_Dispersant = 100503
    Dredge_Rate = 100504
    Dredge_Spill = 100505
    Clearness_Coefficient = 100506
    Coastline_Orientation = 100507
    Reduction_Factor = 100508
    Active_Beach_Height = 100509
    Update_Period = 100510
    Accumulated_Erosion = 100511
    Erosion_Rate = 100512
    Non_dimensional_Transport = 100513
    Local_coordinate = 100514
    Radii_of_Gyration = 100515
    Percentage = 100516
    Line_Capacity = 100517
    Undefined = 999
    Diverted_discharge = 110001
    Demand_carry_over_fraction = 110002
    Groundwater_demand = 110003
    Dam_crest_level = 110004
    Seepage_flux = 110005
    Seepage_fraction = 110006
    Evaporation_fraction = 110007
    Residence_time = 110008
    Owned_fraction_of_inflow = 110009
    Owned_fraction_of_volume = 110010
    Reduction_level = 110011
    Reduction_threshold = 110012
    Reduction_fraction = 110013
    Total_losses = 110014
    Counts_per_liter = 110015
    Assimilative_Capacity = 110016
    Still_Water_Depth = 110017
    Total_Water_Depth = 110018
    Maximum_wave_height = 110019
    Ice_concentration = 110020
    Wind_friction_speed = 110021
    Roughness_length = 110022
    Drag_coefficient = 110023
    Charnock_constant = 110024
    Breaking_parameter_Gamma = 110025
    Threshold_period = 110026
    Courant_number = 110027
    Time_step_factor = 110028
    Element_length = 110029
    Element_area = 110030
    Roller_angle = 110031
    Rate_of_bed_level_change = 110032
    Bed_level_change = 110033
    Sediment_transport_direction = 110034
    Wave_action_density = 110035
    Zero_moment_of_wave_action = 110036
    First_moment_of_wave_action = 110037
    Bed_Mass = 110038
    Water_Quality = 110039
    Status = 110040
    Setting = 110041
    Reaction_Rate = 110042
    Fast_Runoff_Discharge = 110043
    Slow_Runoff_Discharge = 110044
    Average_Sediment_Transport_per_length_unit = 110045
    Valve_Setting = 110046
    Wave_energy_density = 110047
    Wave_energy_distribution = 110048
    Wave_energy = 110049
    Radiation_Melting_Coefficient = 110050
    Rain_melting_coefficient = 110051
    Friction = 110052
    Wave_action_density_rate = 110053
    Element_area_long_lat = 110054
    Electric_Current = 110100
    Heat_Flux_Resistance = 110200
    Absolute_Humidity = 110210
    Length = 110220
    Area = 110225
    Volume = 110230
    Element_Volume = 110231
    Wave_Power = 110232
    Moment_of_Inertia = 110233
    Topography = 110234
    Scour_Depth = 110235
    Scour_Width = 110236
    Cost_per_Volume = 110237
    Cost_per_Energy = 110238
    Cost_per_Mass = 110239
    Application_Intensity = 110240
    Cost = 110241
    Voltage = 110242
    Normal_Velocity = 110243
    Gravity = 110244
    Vessel_Displacement = 110245
    Hydrostatic_Matrix = 110246
    Wave_Number = 110247
    Radiation_Potential = 110248
    Added_Mass_TT = 110249
    Radiation_Damping = 110250
    Frequency = 110251
    Sound_Exposure_Level = 110252
    Transmission_Loss = 110253
    pH = 110254
    Acoustic_Attenuation = 110255
    Sound_Speed = 110256
    Leakage = 110257
    Height_Above_Keel = 110258
    Submerged_Mass = 110259
    Deflection = 110260
    Linear_Damping_Coefficient = 110261
    Quadratic_Damping_Coefficient = 110262
    Damping_TT = 110263
    RAO_Motion = 110264
    RAO_Rotation = 110265
    Added_Mass_Coefficient = 110266
    Electrical_Conductivity = 110267
    Added_Mass_TR = 110268
    Added_Mass_RT = 110269
    Added_Mass_RR = 110270
    Damping_TR = 110271
    Damping_RT = 110272
    Damping_RR = 110273
    Fender_Force = 110274
    Force = 110275
    Moment = 110276
    Reduced_Pollutant_Load = 110277
    Size_and_Position = 110278
    Frame_Rate = 110279
    Dynamic_Viscosity = 110280
    Grid_Rotation = 110281
    Agent_Density = 110282
    Emitter_Coefficient = 110283
    Pipe_Diameter = 110284
    Speed = 110285
    Velocity = 110286
    Direction = 110287
    Displacement = 110288
    Position = 110289
    Rotation = 110290
    Torque = 110291
    Overtopping = 110292
    Flow_Rate = 110293
    Acceleration = 110294
    Dimensionless_Acceleration = 110295
    Time = 110296
    Resistance = 110297
    Amount_of_Substance = 110298
    Molar_Concentration = 110299
    Molal_Concentration = 110300
    Suspended_sediment_load_per_area = 110301
    Bollard_Force = 110302
    Discharge_per_Pressure = 110303
    RotationalSpeed = 110304
    Infiltration_per_Area = 110305
    Mass_per_Length_per_Time = 110306
    NearBedLoad_per_Length = 110307
    Substance_per_UnitArea = 110308
    AccNearBedLoad_per_Length = 110309
    ThermalConductivity = 110310
    DirectionalVariance = 110311
    SpecificDissipationRate = 110312
    AngularFrequency = 110313
    StemDiameter = 110314
    VegetationDensity = 110315
    ElasticModulus = 110316
    BladeWidth = 110317
    BladeThickness = 110318
    PlantDensity = 110319
    Thickness = 110320

    def __init__(self, code: int) -> None:
        self.code = code

    @property
    def display_name(self) -> str:
        """Display friendly name."""
        name = self.name
        name = name.replace("_", " ")
        return name

    def __repr__(self) -> str:
        return self.display_name

    @property
    def units(self) -> list[EUMUnit]:
        """List valid units for this EUM type."""
        temp = _unit_list(self.code).values()
        return [EUMUnit(value) for value in temp]

    @staticmethod
    def search(pattern: str) -> list[EUMType]:
        temp = _type_list(pattern).keys()
        return [EUMType(key) for key in temp]


class EUMUnit(IntEnum):
    """EUM unit.

    Examples
    --------
    ```{python}
    import mikeio
    mikeio.EUMUnit.degree_Kelvin
    ```

    """

    meter = 1000
    kilometer = 1001
    centimeter = 1007
    millimeter = 1002
    feet = 1003
    feet_US = 1014
    inch = 1004
    inch_US = 1013
    mile = 1005
    mile_US = 1016
    yard = 1006
    yard_US = 1015
    meter_pow_3_per_sec = 1800
    meter_pow_3_per_min = 1816
    feet_pow_3_per_sec = 1801
    feet_pow_3_per_min = 1817
    feet_pow_3_per_day = 1814
    feet_pow_3_per_year = 1815
    meter_pow_3_per_day = 1810
    meter_pow_3_per_year = 1805
    acre_feet_per_day = 1804
    meter_pow_3_per_hour = 1821
    gallon_per_min = 1818
    liter_per_minute = 1820
    liter_per_sec = 1819
    Mgal_per_day = 1803
    MgalUK_per_day = 1823
    Ml_per_day = 1802
    meter_per_sec = 2000
    feet_per_sec = 2002
    miles_per_hour = 2020
    km_per_hour = 2021
    knot = 2019
    degree = 2401
    degree_Celsius = 2800
    delta_degree_Celsius = 2900
    degree_Fahrenheit = 2801
    delta_degree_Fahrenheit = 2901
    degree_Kelvin = 2802
    mu_g_per_meter_pow_3 = 2201
    mg_per_meter_pow_3 = 2202
    gram_per_meter_pow_3 = 2203
    kg_per_meter_pow_3 = 2200
    mu_g_per_liter = 2204
    mg_per_liter = 2205
    gram_per_liter = 2206
    pound_per_feet_pow_3 = 2207
    ton_per_meter_pow_3 = 2208
    million_per__100_ml = 3000
    per__100_ml = 3001
    per_liter = 3002
    __ = 99000
    feet_US_pow_3_per_sec = 1831
    feet_US_pow_3_per_day = 1833
    feet_US_pow_3_per_year = 1834
    feet_US_pow_3_per_min = 1832
    yard_US_pow_3_per_sec = 1835
    yard_pow_3_per_sec = 1830
    meter_per_day = 2006
    feet_US_per_sec = 2036
    percent = 99001
    percent_per_day = 2601
    cm_per_sec = 2023
    mm_per_sec = 2017
    inch_per_sec = 2011
    inch_US_per_sec = 2035
    hours_per_day = 99003
    watt_per_meter_pow_2 = 6600
    joule_per_meter_pow_2_per_day = 5703
    kJ_per_meter_pow_2_per_hour = 5700
    kJ_per_meter_pow_2_per_day = 5701
    MJ_per_meter_pow_2_per_day = 5702
    PSU = 6200
    per_thousand = 99002
    meter_pow_2 = 3200
    feet_pow_2 = 3203
    meter_pow__1_per_3__per_sec = 3600
    feet_pow__1_per_3__per_sec = 3601
    sec_per_meter_pow__1_per_3_ = 3800
    sec_per_feet_pow__1_per_3_ = 3801
    meter_pow__1_per_2__per_sec = 4000
    feet_pow__1_per_2__per_sec = 4001
    feet_US_pow__1_per_2__per_sec = 4002
    meter_pow_3 = 1600
    liter = 1601
    megaliter = 1609
    km_pow_3 = 1606
    acre_feet = 1607
    feet_pow_3 = 1603
    gallon = 1604
    megagallon = 1608
    _10_pow_6meter_pow_3 = 1610
    acre = 3202
    kilogram = 1200
    gram = 1201
    milligram = 1202
    microgram = 1203
    ton = 1204
    pound = 1207
    kiloton = 1205
    megaton = 1206
    ounce = 1209
    per_kg = 1250
    per_gram = 1251
    per_mg = 1252
    per_mu_g = 1253
    per_ton = 1254
    per_kiloton = 1255
    per_megaton = 1256
    per_pound = 1257
    per_ton_US = 1258
    per_ounce = 1259
    kg_per_sec = 4200
    gram_per_sec = 4203
    mg_per_sec = 4202
    mu_g_per_sec = 4201
    pound_per_sec = 4215
    yard_pow_3 = 1615
    yard_US_pow_3 = 1614
    liter_per_day = 1836
    mm_per_hour = 2001
    mm_per_day = 2004
    inch_per_hour = 2016
    cm_per_hour = 2018
    feet_per_day = 2009
    inch_per_min = 2014
    mu_m_per_sec = 2031
    millimeter_per_day = 4801
    inch_per_day = 4802
    liter_per_sec_per_ha = 2030
    mm_per_year = 2040
    meter_pow_3_per_meter = 3201
    feet_pow_3_per_feet = 3207
    feet_US_pow_3_per_feet_US = 3214
    yard_pow_3_per_yard = 3213
    yard_US_pow_3_per_yard_US = 3212
    kg_per_hour = 4204
    gram_per_meter_pow_2 = 4400
    gram_per_meter_pow_2_per_day = 4500
    gram_per_meter_pow_3_per_hour = 4600
    ton_US = 1208
    kg_per_meter = 4401
    meter_pow_3_per_sec_per_meter = 4700
    meter_pow_2_per_sec = 4702
    feet_pow_2_per_sec = 4704
    cm_pow_3_per_sec_per_cm = 4721
    mm_pow_3_per_sec_per_mm = 4722
    feet_pow_3_per_sec_per_feet = 4718
    feet_US_pow_3_per_sec_per_feet_US = 4723
    inch_pow_3_per_sec_per_inch = 4724
    inch_US_pow_3_per_sec_per_inch_US = 4725
    yard_US_pow_3_per_sec_per_yard_US = 4726
    yard_pow_3_per_sec_per_yard = 4727
    meter_pow_3_per_year_per_meter = 4701
    second = 1400
    hertz = 2602
    day = 1403
    liter_per_sec_per_km_pow_2 = 2003
    acre_feet_per_sec_per_acre = 2005
    feet_pow_3_per_sec_per_mile_pow_2 = 2007
    watt = 4900
    kilowatt = 4901
    megawatt = 4902
    gigawatt = 4903
    radian = 2400
    per_meter = 5000
    per_feet = 5003
    per_feet_US = 5005
    meter_pow_3_per_sec_pow_2 = 6800
    meter_pow_2_sec_per_rad = 5200
    feet_pow_2_sec_per_rad = 5213
    meter_pow_2_per_rad = 5201
    feet_pow_2_per_rad = 5215
    meter_pow_2_sec = 5202
    feet_pow_2_sec = 5217
    square_meter_per_sec = 9400
    square_feet_per_sec = 9401
    Integer = 99013
    hectare = 3204
    km_pow_2 = 3205
    mile_pow_2 = 3206
    millifeet = 1010
    kg_per_day = 4205
    kg_per_year = 4207
    hour = 1402
    per_hour = 2603
    per_day = 2600
    per_sec = 2605
    gram_per_meter_pow_2_per_sec = 4501
    kg_per_meter_pow_2_per_sec = 4503
    newton_per_meter_pow_2 = 5400
    kN_per_meter_pow_2 = 5401
    pound_per_feet_per_sec_pow_2 = 5402
    newton_per_meter_pow_3 = 5500
    kN_per_meter_pow_3 = 5501
    kilogram_M2 = 5550
    poundSqrFeet = 5551
    gram_per_meter_pow_3_per_day = 4601
    _mg_per_l__pow__1_per_2__per_day = 5300
    _mg_per_l__pow__1_per_2__per_hour = 5301
    meter_per_hour = 2008
    gal_per_day_per_head = 1806
    liter_per_day_per_head = 1807
    meter_pow_3_per_sec_per_head = 1808
    mm_per_month = 2010
    mm_per_C_per_day = 5800
    mm_per_C_per_hour = 5801
    inch_per_F_per_day = 5802
    inch_per_F_per_hour = 5803
    _per_C_per_day = 5900
    _per_C_per_hour = 5901
    _per_F_per_day = 5902
    _per_F_per_hour = 5903
    Celsius_per__100meter = 6000
    Celsius_per__100feet = 6001
    Fahrenheit_per__100feet = 6003
    Fahrenheit_per__100meter = 6002
    percent_per__100meter = 5001
    percent_per__100feet = 5002
    gram_per_day = 4206
    ton_per_sec = 4220
    pascal = 6100
    hectopascal = 6101
    millibar = 6108
    meter_pow_2_per_sec_pow_2 = 6400
    feet_pow_2_per_sec_pow_2 = 4720
    meter_pow_2_per_sec_pow_3 = 6401
    feet_pow_2_per_sec_pow_3 = 6402
    PSU_meter_pow_3_per_sec = 6300
    non_dim_meter_pow_3_per_sec = 6302
    PSU_feet_pow_3_per_sec = 6303
    C_meter_pow_3_per_sec = 6301
    F_feet_pow_3_per_sec = 6304
    mg_per_gram = 99007
    ml_per_liter = 99018
    per_million = 99020
    mg_per_kg = 99008
    mu_l_per_liter = 99019
    joule_per_kg = 6500
    joule_kg_per_K = 6700
    meter_per_min = 2012
    feet_per_min = 2013
    feet_per_hour = 2015
    feet_pow_3_per_sec_pow_2 = 6801
    nautical_mile = 1009
    minute = 1401
    month = 1405
    year = 1404
    gram_per_meter_pow_3_per_sec = 4602
    meter_pow_3_per_gram = 6900
    liter_per_gram = 6901
    meter_pow_3_per_ha_per_sec = 2027
    meter_pow_3_per_ha_per_hour = 2026
    meter_pow_3_per_ha_per_day = 2025
    feet_pow_3_per_acre_per_sec = 2024
    feet_pow_3_per_acre_per_hour = 2041
    feet_pow_3_per_acre_per_day = 2042
    liter_per_min_per_ha = 2029
    gallon_per_min_per_acre = 2028
    horsepower = 4904
    kg_per_ha_per_hour = 4502
    kg_per_ha_per_day = 4504
    pound_per_feet_pow_2_per_sec = 4507
    pound_per_acre_per_day = 4505
    newton = 7000
    meter_pow_2_per_hertz = 7100
    feet_pow_2_per_hertz = 7103
    meter_pow_2_per_hertz_per_deg = 7101
    meter_pow_2_per_hertz_per_rad = 7102
    feet_pow_2_per_hertz_per_deg = 7104
    feet_pow_2_per_hertz_per_rad = 7105
    kg_per_sec_pow_2 = 8100
    meter_pow_2_per_kg = 9100
    person = 99004
    currency_per_year = 2604
    liter_per_person_per_day = 1809
    meter_pow_3_per_currency = 1611
    meter_pow_3_per_km_pow_2_per_day = 4803
    per_inch = 5004
    meter_pow_2_per_hour = 4708
    meter_pow_2_per_day = 4709
    feet_pow_2_per_hour = 4710
    feet_pow_2_per_day = 4711
    feet_US_per_day = 2037
    inch_US_per_hour = 2038
    inch_US_per_min = 2039
    gallon_per_sec = 1811
    gallon_per_day = 1812
    gallon_per_year = 1813
    kg_per_meter_pow_2 = 4402
    meter_pow_3_per_meter_pow_3 = 99011
    liter_per_meter_pow_3 = 99012
    kg_per_ha = 4403
    kg_per_km_pow_2 = 4406
    gram_per_km_pow_2 = 4408
    gram_per_ha = 4410
    ton_per_meter_pow_2 = 2210
    ton_per_km_pow_2 = 4407
    ton_per_ha = 4409
    pound_per_feet_pow_2 = 2209
    pound_per_acre = 4405
    pound_per_mile_pow_2 = 4411
    kg_per_acre = 4412
    kg_per_feet_pow_2 = 4413
    kg_per_mile_pow_2 = 4414
    ton_per_acre = 4415
    ton_per_feet_pow_2 = 4416
    ton_per_mile_pow_2 = 4417
    gram_per_acre = 4418
    gram_per_feet_pow_2 = 4419
    gram_per_mile_pow_2 = 4420
    pound_per_ha = 4421
    pound_per_meter_pow_2 = 4422
    pound_per_km_pow_2 = 4423
    pound_per_feet_US_pow_3 = 2214
    pound_per_yard_US_pow_3 = 2212
    pound_per_yard_pow_3 = 2213
    gram_per_gram = 99005
    gram_per_kg = 99006
    mu_g_per_gram = 99009
    kg_per_kg = 99010
    micrometer = 1008
    _10_pow_9_per_day = 2606
    _10_pow_12_per_year = 2607
    meter_pow_3_per_sec_per__10mm = 4706
    feet_pow_3_per_sec_per_inch = 4707
    mg_per_meter_pow_2 = 4404
    mu_g_per_meter_pow_2 = 2211
    _1000_per_meter_pow_2_per_day = 3401
    _per_meter_pow_2_per_sec = 3402
    terajoule = 5604
    gigajoule = 5603
    megajoule = 5602
    kilojoule = 5601
    joule = 5600
    petajoule = 5607
    exajoule = 5608
    kilowatt_hour = 5605
    megawatt_hour = 5609
    gigawatt_hour = 5610
    liter_per_meter_pow_2 = 1011
    meter_pow_3_per_hour_per_meter = 7503
    meter_pow_3_per_day_per_meter = 7504
    feet_pow_3_per_hour_per_feet = 4719
    m3_per_hour_per_M = 4730
    m3_per_day_per_M = 4731
    feet_pow_3_per_day_per_feet = 4732
    galUK_per_day_per_feet = 4712
    gallon_per_day_per_feet = 4713
    gallon_per_min_per_feet = 4714
    liter_per_day_per_meter = 4715
    liter_per_minute_per_meter = 4716
    liter_per_second_per_meter = 4717
    liter_per_min_per_meter = 7501
    liter_per_sec_per_meter = 7500
    gram_per_min = 4208
    pound_per_day = 4212
    pound_per_hour = 4213
    pound_per_min = 4214
    pound_per_year = 4217
    ton_per_year = 4218
    MetresWater = 6105
    FeetWater = 6106
    kilopascal = 6102
    megapascal = 6104
    psi = 6103
    bar = 6107
    decibar = 6110
    gigapascal = 6111
    kg_per_meter_pow_2_per_day = 4506
    meter_pow_3_per_mg = 6902
    meter_pow_3_per_mu_g = 6903
    ton_per_day = 4219
    millimeterD50 = 1012
    millisecond = 1406
    feet_US_pow_2 = 3208
    yard_US_pow_2 = 3209
    mile_US_pow_2 = 3210
    acre_US = 3211
    liter_per_meter = 3215
    milliliter = 1602
    milligallon = 1605
    gallonUK = 1612
    megagallonUK = 1613
    galUK_per_day = 1822
    feet_pow_3_per_PE_per_day = 1824
    meter_pow_3_per_PE_per_day = 1825
    galUK_per_sec = 1826
    galUK_per_year = 1827
    galUK_per_PE_per_day = 1828
    ydUS3_per_sec = 1829
    acre_feet_per_day_per_acre = 2022
    Mgal_per_day_per_acre = 2032
    MgalUK_per_day_per_acre = 2033
    Ml_per_day_per_ha = 2034
    liter_per_hour_per_ha = 2043
    liter_per_day_per_ha = 2044
    meter_per_sec_pow_2 = 2100
    feet_per_sec_pow_2 = 2101
    pound_per_feet_US_pow_2 = 2215
    ounce_per_cubic_feet = 2216
    ounce_per_cubic_feet_US = 2217
    ounce_per_Yard3 = 2218
    ounce_per_yard_US3 = 2219
    ounce_per_square_feet = 2220
    ounce_per_square_feet_US = 2221
    gramPerCubicCentimeter = 2222
    kg_per_meter_per_sec = 2300
    Pascal_second = 2301
    kilogram_per_meter_per_day = 2302
    gram_per_Meter_per_Day = 2303
    gram_per_Km_per_Day = 2304
    pound_per_Feet_per_Day = 2305
    pound_per_FeetUS_per_Day = 2306
    ounce_per_Feet_per_Day = 2307
    ounce_per_FeetUS_per_Day = 2308
    kilogram_per_Yard_per_Second = 2309
    kilogram_per_Feet_per_Second = 2310
    pound_per_Yard_per_Second = 2311
    pound_per_Feet_per_Second = 2312
    degree50 = 2402
    degree_pow_2 = 2403
    radian_pow_2 = 2404
    degree_per_meter = 2500
    radian_per_meter = 2501
    degree_per_sec = 2510
    radian_per_sec = 2511
    meter_pow_2_per_sec_per_ha = 2608
    feet_pow_2_per_sec_per_acre = 2609
    rev_per_min = 2610
    percent_per_hour = 2611
    percent_per_sec = 2613
    revolution_per_second = 2614
    revolution_per_hour = 2615
    _per_degree_C = 2850
    _per_degree_F = 2851
    per_meter_pow_3 = 3003
    per_ml = 3004
    per_feet_pow_3 = 3005
    per_gal = 3006
    per_mgal = 3007
    per_km_pow_3 = 3008
    per_acre_feet = 3009
    per_Mgal = 3010
    per_Ml = 3011
    per_gallonUK = 3012
    per_MgalUK = 3013
    per_yard_US_pow_3 = 3014
    per_yard_pow_3 = 3015
    sec_per_meter = 3100
    Einstein_per_meter_pow_2_per_day = 3400
    kg_per_PE_per_day = 4209
    kg_per_min = 4210
    pound_per_PE_per_day = 4216
    mg_per_ha = 4424
    mg_per_km_pow_2 = 4425
    mg_per_acre = 4426
    mg_per_feet_pow_2 = 4427
    mg_per_mile_pow_2 = 4428
    pound_per_meter = 4429
    ton_per_meter = 4430
    pound_per_Feet = 4431
    pound_per_Yard = 4432
    pound_per_FeetUS = 4433
    pound_per_YardUS = 4434
    ounce_per_Feet = 4435
    ounce_per_Yard = 4436
    ounce_per_FeetUS = 4437
    ounce_per_YardUS = 4438
    kilogram_per_Yard = 4439
    kilogram_per_Feet = 4440
    mg_per_liter_per_day = 4603
    yard_pow_3_per_year_per_yard = 4728
    yard_US_pow_3_per_year_per_yard_US = 4729
    per_inch_US = 5006
    meter_pow_3_per_sec_2 = 5100
    meter_pow_2_per_deg = 5203
    meter_pow_2_sec_pow_2_per_rad = 5204
    meter_pow_2_per_sec_per_rad = 5205
    meter_pow_2_sec_per_deg = 5206
    meter_pow_2_sec_pow_2_per_deg = 5207
    meter_pow_2_per_sec_per_deg = 5208
    feet_pow_2_per_sec_per_rad = 5209
    feet_pow_2_per_sec_per_deg = 5210
    feet_pow_2_sec_pow_2_per_rad = 5211
    feet_pow_2_sec_pow_2_per_deg = 5212
    feet_pow_2_sec_per_deg = 5214
    feet_pow_2_per_deg = 5216
    kg_meter_pow_2 = 7060
    watt_second = 5606
    per_joule = 5650
    per_kJ = 5651
    per_MJ = 5652
    per_GJ = 5653
    per_TJ = 5654
    per_PJ = 5655
    per_EJ = 5656
    per_kWh = 5657
    per_Ws = 5658
    per_MWh = 5659
    per_GWh = 5660
    mm_per__kJ_per_meter_pow_2_ = 5710
    mm_per__MJ_per_meter_pow_2_ = 5711
    micropascal = 6109
    dB_re_1_mu_Pa_pow_2_sec = 6150
    dB_per_lambda = 6160
    meter_pow_2_per_sec_pow_3_per_rad = 6403
    feet_pow_2_per_sec_pow_3_per_rad = 6404
    AFD_per_sec = 6802
    IMGD_per_sec = 6803
    MGD_per_sec = 6804
    GPM_per_sec = 6805
    meter_pow_3_per_day_per_sec = 6806
    meter_pow_3_per_hour_per_sec = 6807
    Ml_per_day_per_sec = 6808
    liter_per_min_per_sec = 6809
    liter_per_sec_pow_2 = 6810
    kilonewton = 7001
    meganewton = 7002
    millinewton = 7003
    kg_meter = 7050
    kg_meter_per_sec = 7070
    kg_meter_pow_2_per_sec = 7080
    meter_pow_2_per_hertz_pow_2 = 7200
    meter_pow_2_per_hertz_pow_2_per_deg = 7201
    meter_pow_2_per_hertz_pow_2_per_rad = 7202
    Ml_per_day_per_meter = 7502
    feet_pow_3_per_sec_per_psi = 7505
    gallon_per_min_per_psi = 7506
    Mgal_per_day_per_psi = 7507
    MgalUK_per_day_per_psi = 7508
    acre_feet_per_day_per_psi = 7509
    m3_per_hour_per_bar = 7510
    _per_meter_per_sec = 9200
    meter_per_sec_per_ha = 9201
    feet_per_sec_per_acre = 9202
    per_meter_pow_2 = 9300
    per_acre = 9301
    per_hectare = 9302
    per_km_pow_2 = 9303
    per_square_feet = 9304
    per_cubic_meter = 9350
    currency_per_meter_pow_3 = 9351
    currency_per_feet_pow_3 = 9352
    per_watt = 9600
    newton_meter = 9700
    kilonewton_meter = 9701
    meganewton_meter = 9702
    newton_millimeter = 9703
    newton_meter_second = 9800
    newton_per_meter_per_sec = 9900
    mole = 12000
    millimole = 12001
    micromole = 12002
    nanomole = 12003
    mole_per_liter = 12020
    mmol_per_liter = 12021
    mu_mol_per_liter = 12022
    nmol_per_liter = 12023
    mole_per_meter_pow_3 = 12024
    mmol_per_meter_pow_3 = 12025
    mu_mol_per_meter_pow_3 = 12026
    mole_per_kg = 12040
    mmol_per_kg = 12041
    mu_mol_per_kg = 12042
    nmol_per_kg = 12043
    mole_per_m2 = 12060
    millimole_per_m2 = 12061
    micromole_per_m2 = 12062
    nanomole_per_m2 = 12063
    meter_per_meter = 99014
    per_minute = 99015
    percent_per_min = 2612
    per_month = 99016
    per_year = 99017
    g_acceleration = 99021
    ampere = 99100
    milliampere = 99101
    microampere = 99102
    kiloampere = 99103
    megaampere = 99104
    volt = 99150
    millivolt = 99151
    microvolt = 99152
    kilovolt = 99153
    megavolt = 99154
    ohm = 99180
    kOhm = 99181
    Mohm = 99182
    undefined = 0
    watt_per_meter = 99200
    kW_per_meter = 99201
    MW_per_meter = 99202
    GW_per_meter = 99203
    kW_per_feet = 99204
    watt_per_meter_per_degree_celsius = 99220
    watt_per_feet_per_degree_fahrenheit = 99221
    siemens = 99250
    millisiemens = 99251
    microsiemens = 99252
    siemens_per_meter = 99260
    mS_per_cm = 99261
    mu_S_per_cm = 99262
    kg_per__meter_sec_ = 99263
    cPoise = 99264
    lbf_sec_per_feet_pow_2 = 99265
    pound_per__sec_feet_ = 99266

    def __init__(self, code: int) -> None:
        self.code = code

    @property
    def display_name(self) -> str:
        """Display friendly name."""
        name = self.name
        name = name.replace("_", " ")
        return name

    @property
    def short_name(self) -> str:
        unit_short_names = {
            "kilometer": "km",
            "centimeter": "cm",
            "millimeter": "mm",
            "meter": "m",
            "liter": "l",
            "kilogram": "kg",
            "milligram": "mg",
            "gram": "g",
            "second": "s",
            "sec": "s",
            "hertz": "Hz",
            "_pow_2": "^2",
            "_pow_3": "^3",
            "_per_": "/",
            "_": "*",
        }

        name = self.name
        for key, value in unit_short_names.items():
            name = name.replace(key, value)
        return name

    def __repr__(self) -> str:
        return self.display_name


class ItemInfo:
    """Info for dynamicc items (variables).

    Parameters
    ----------
    name: str or EUMType, optional
    type: EUMType or int, optional
        Default EUMType.Undefined
    unit: EUMUnit or int, optional
        Default unit matching EUMType
    data_value_type: str, optional
        One of the following strings: 'Instantaneous', 'Accumulated', 'StepAccumulated', 'MeanStepBackward',
        'MeanStepForward'. Default: 'Instantaneous'

    Examples
    --------
    ```{python}
    import mikeio
    mikeio.ItemInfo("Viken", mikeio.EUMType.Water_Level)
    ```

    ```{python}
    mikeio.ItemInfo(mikeio.EUMType.Wind_speed)
    ```

    """

    def __init__(
        self,
        name: str | EUMType | None = None,
        itemtype: EUMType | EUMUnit | None = None,
        unit: EUMUnit | None = None,
        data_value_type: Literal[
            "Instantaneous", "Accumulated", "StepAccumulated", "MeanStepBackward"
        ] = "Instantaneous",
    ) -> None:
        # Handle arguments in the wrong place
        if isinstance(name, EUMType):
            if isinstance(itemtype, EUMUnit):
                unit = itemtype

            itemtype = name
            name = name.display_name

        if itemtype is not None:
            if isinstance(itemtype, int):
                itemtype = EUMType(itemtype)

            if not isinstance(itemtype, EUMType):
                raise ValueError(
                    "Invalid type. Type should be supplied as EUMType, e.g. ItemInfo('WL',EUMType.Water_Level, EUMUnit.meter)"
                )
            self.type = itemtype

            if name is None:
                name = itemtype.display_name
        else:
            self.type = EUMType.Undefined

        if unit is not None:
            if isinstance(unit, int):
                unit = EUMUnit(unit)

            if not isinstance(unit, EUMUnit):
                raise ValueError(
                    "Invalid unit. Unit should be supplied as EUMUnit, e.g. ItemInfo('WL',EUMType.Water_Level, EUMUnit.meter)"
                )
            self._unit = unit
        else:
            if self.type == EUMType.Undefined:
                self._unit = EUMUnit.undefined
            else:
                self._unit = self.type.units[0]

        self.data_value_type = to_datatype(data_value_type)

        if not isinstance(name, str):
            raise ValueError("Invalid name, name should be a string")
        self.name: str = name

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ItemInfo):
            return NotImplemented

        # an alternative approach to this long expression is to use dataclasses (Python>=3.7)
        return (
            self.name == other.name
            and self.type == other.type
            and self.unit == other.unit
            and self.data_value_type == other.data_value_type
        )

    def __repr__(self) -> str:
        if self.data_value_type == DataValueType.Instantaneous:
            return f"{self.name} <{self.type.display_name}> ({self.unit.display_name})"
        else:
            return f"{self.name} <{self.type.display_name}> ({self.unit.display_name}) - {self.data_value_type}"

    @property
    def unit(self) -> EUMUnit:
        "Item unit."
        return self._unit

    @unit.setter
    def unit(self, value: EUMUnit) -> None:
        "Set unit."
        if value not in self.type.units:
            raise ValueError(
                f"{value} is not a correct unit for {self.type}. Use {self.type.units}"
            )
        self._unit = value

    @staticmethod
    def from_mikecore_dynamic_item_info(dfsItemInfo: DfsDynamicItemInfo) -> "ItemInfo":
        """Create ItemInfo from a mikecore.DfsDynamicItemInfo object."""
        name = dfsItemInfo.Name
        item = dfsItemInfo.Quantity.Item
        unit = dfsItemInfo.Quantity.Unit
        itemtype = EUMType(item)
        unit = EUMUnit(unit)
        data_value_type = dfsItemInfo.ValueType

        return ItemInfo(name, itemtype, unit, data_value_type)


class ItemInfoList(list):
    def __init__(self, items: Sequence[ItemInfo]):
        super().__init__(items)

    def to_dataframe(self) -> pd.DataFrame:
        data = [
            {"name": item.name, "type": item.type.name, "unit": item.unit.name}
            for item in self
        ]
        return pd.DataFrame(data)


def to_datatype(datatype: str | int | DataValueType) -> DataValueType:
    string_datatype_mapping = {
        "Instantaneous": DataValueType.Instantaneous,
        "Accumulated": DataValueType.Accumulated,
        "StepAccumulated": DataValueType.StepAccumulated,
        "MeanStepBackward": DataValueType.MeanStepBackward,
        "MeanStepForward": DataValueType.MeanStepForward,
        0: DataValueType.Instantaneous,
        1: DataValueType.Accumulated,
        2: DataValueType.StepAccumulated,
        3: DataValueType.MeanStepBackward,
        4: DataValueType.MeanStepForward,
    }

    if isinstance(datatype, str):
        if datatype not in string_datatype_mapping.keys():
            raise InvalidDataValueType

        return string_datatype_mapping[datatype]

    if isinstance(datatype, int):
        if datatype not in string_datatype_mapping.keys():
            raise InvalidDataValueType

        return string_datatype_mapping[datatype]

    if not isinstance(datatype, DataValueType):
        raise ValueError("Data value type not supported")

    return datatype
