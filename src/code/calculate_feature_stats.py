import torch

# Assuming fileList is a list containing the filenames
fileList =  ["features_batch_0_1999.pt",    "features_batch_0_2999.pt",    "features_batch_0_3999.pt",    "features_batch_0_4999.pt",    "features_batch_0_5999.pt",    "features_batch_0_6999.pt",    "features_batch_0_7999.pt",    "features_batch_0_8999.pt",    "features_batch_0_9.pt",       "features_batch_0_999.pt",     "features_batch_1_10999.pt",   "features_batch_1_11999.pt",   "features_batch_1_12999.pt",   "features_batch_1_13999.pt",   "features_batch_1_14999.pt",   "features_batch_1_15999.pt",   "features_batch_1_16999.pt",   "features_batch_1_17999.pt",   "features_batch_1_18999.pt",   "features_batch_1_9999.pt",    "features_batch_10_100999.pt", "features_batch_10_101999.pt", "features_batch_10_102999.pt", "features_batch_10_103999.pt", "features_batch_10_104999.pt", "features_batch_10_105999.pt", "features_batch_10_106999.pt", "features_batch_10_107999.pt", "features_batch_10_108999.pt", "features_batch_10_99999.pt",  "features_batch_11_109999.pt", "features_batch_11_110999.pt", "features_batch_11_111999.pt", "features_batch_11_112999.pt", "features_batch_11_113999.pt", "features_batch_11_114999.pt", "features_batch_11_115999.pt", "features_batch_11_116999.pt", "features_batch_11_117999.pt", "features_batch_11_118999.pt", "features_batch_12_119999.pt", "features_batch_12_120999.pt", "features_batch_12_121999.pt", "features_batch_12_122999.pt", "features_batch_12_123999.pt", "features_batch_12_124999.pt", "features_batch_12_125999.pt", "features_batch_12_126999.pt", "features_batch_12_127999.pt", "features_batch_12_128999.pt", "features_batch_13_129999.pt", "features_batch_13_130999.pt", "features_batch_13_131999.pt", "features_batch_13_132999.pt", "features_batch_13_133999.pt", "features_batch_13_134999.pt", "features_batch_13_135999.pt", "features_batch_13_136999.pt", "features_batch_13_137999.pt", "features_batch_13_138999.pt", "features_batch_14_139999.pt", "features_batch_14_140999.pt", "features_batch_14_141999.pt", "features_batch_14_142999.pt", "features_batch_14_143999.pt", "features_batch_14_144999.pt", "features_batch_14_145999.pt", "features_batch_14_146999.pt", "features_batch_14_147999.pt", "features_batch_2_19999.pt",   "features_batch_2_20999.pt",   "features_batch_2_21999.pt",   "features_batch_2_22999.pt",   "features_batch_2_23999.pt",   "features_batch_2_24999.pt",   "features_batch_2_25999.pt",   "features_batch_2_26999.pt",   "features_batch_2_27999.pt",   "features_batch_2_28999.pt",   "features_batch_3_29999.pt",   "features_batch_3_30999.pt",   "features_batch_3_31999.pt",   "features_batch_3_32999.pt",   "features_batch_3_33999.pt",   "features_batch_3_34999.pt",   "features_batch_3_35999.pt",   "features_batch_3_36999.pt",   "features_batch_3_37999.pt",   "features_batch_3_38999.pt",   "features_batch_4_39999.pt",   "features_batch_4_40999.pt",   "features_batch_4_41999.pt",   "features_batch_4_42999.pt",   "features_batch_4_43999.pt",   "features_batch_4_44999.pt",   "features_batch_4_45999.pt",   "features_batch_4_46999.pt",   "features_batch_4_47999.pt",   "features_batch_4_48999.pt",   "features_batch_5_49999.pt",   "features_batch_5_50999.pt",   "features_batch_5_51999.pt",   "features_batch_5_52999.pt",   "features_batch_5_53999.pt",   "features_batch_5_54999.pt",   "features_batch_5_55999.pt",   "features_batch_5_56999.pt",   "features_batch_5_57999.pt",   "features_batch_5_58999.pt",   "features_batch_6_59999.pt", "features_batch_6_60999.pt", "features_batch_6_61999.pt", "features_batch_6_62999.pt", "features_batch_6_63999.pt", "features_batch_6_64999.pt", "features_batch_6_65999.pt", "features_batch_6_66999.pt", "features_batch_6_67999.pt", "features_batch_6_68999.pt", "features_batch_7_69999.pt", "features_batch_7_70999.pt", "features_batch_7_71999.pt", "features_batch_7_72999.pt", "features_batch_7_73999.pt", "features_batch_7_74999.pt", "features_batch_7_75999.pt", "features_batch_7_76999.pt", "features_batch_7_77999.pt", "features_batch_7_78999.pt", "features_batch_8_79999.pt", "features_batch_8_80999.pt", "features_batch_8_81999.pt", "features_batch_8_82999.pt", "features_batch_8_83999.pt", "features_batch_8_84999.pt", "features_batch_8_85999.pt", "features_batch_8_86999.pt", "features_batch_8_87999.pt", "features_batch_8_88999.pt", "features_batch_9_89999.pt", "features_batch_9_90999.pt", "features_batch_9_91999.pt", "features_batch_9_92999.pt", "features_batch_9_93999.pt", "features_batch_9_94999.pt", "features_batch_9_95999.pt", "features_batch_9_96999.pt", "features_batch_9_97999.pt", "features_batch_9_98999.pt"] # List of filenames
# Initialize an empty tensor to accumulate the data
accumulated_data = []

# Load and accumulate the data from the first 10 files
for i, file_to_read in enumerate(fileList):
    # Load the .pt file
    data = torch.load(file_to_read)
    # Accumulate the data
    accumulated_data.append ( data)
#     if i == 9:
#         break
accumulated_data = torch.vstack(accumulated_data)
print("accumulated_data", accumulated_data.shape)
# Print statistics along the first axis
min_values = torch.min(accumulated_data, dim=0)[0]
max_values = torch.max(accumulated_data, dim=0)[0]
mean_values = torch.mean(accumulated_data, dim=0)
std_values = torch.std(accumulated_data, dim=0)

print("Minimum values along the first axis:")
print(min_values)
print("\nMaximum values along the first axis:")
print(max_values)
print("\nMean values along the first axis:")
print(mean_values)
print("\nStandard deviation values along the first axis:")
print(std_values)