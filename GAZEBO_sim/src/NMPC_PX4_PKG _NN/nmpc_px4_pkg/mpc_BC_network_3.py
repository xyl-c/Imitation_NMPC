import torch
from torch import nn


# tensor([ 4.7302e-10, -1.9684e-09, -1.4420e-09,  4.3152e-08,  3.1128e-09,
#          5.2498e-08, -2.6855e-08,  3.5095e-09,  1.8616e-09,  4.1275e-09,
#         -8.5983e-09,  3.1128e-09,  2.9297e-08], device='cuda:0')
# tensor([0.5257, 0.5257, 0.5257, 0.7833, 0.7833, 0.7833, 0.7833, 1.4510, 1.4510,
#         1.4510, 2.3121, 2.3121, 2.3121], device='cuda:0')
# tensor([40.0000,  0.5000,  0.5000,  0.2160], device='cuda:0')
# tensor([ 0.0000, -0.5000, -0.5000, -0.2160], device='cuda:0')      4e6

# tensor([ 5.6704e-09,  1.6200e-09,  5.8369e-10,  3.4445e-08,  3.0268e-09,
#          3.1185e-08, -1.8379e-08, -1.0249e-09, -6.6003e-09, -1.3978e-08,
#         -2.0039e-08, -4.0851e-08,  4.7593e-08])
# tensor([0.4373, 0.4373, 0.4373, 0.7139, 0.7139, 1.0039, 1.0039, 1.3626, 1.3626,
#         1.3626, 2.0662, 2.0662, 2.0662])
# tensor([40.0000,  0.5000,  0.5000,  0.2160])
# tensor([ 0.0000, -0.5000, -0.5000, -0.2160])                        6e6

# tensor([ 4.6043e-09,  4.9973e-10,  2.9297e-09,  2.2614e-08,  3.0632e-09,
#          2.6245e-08, -1.3268e-08,  1.2856e-09,  1.3733e-09, -3.5095e-10,
#         -1.6785e-08, -3.0334e-08,  2.6703e-08], device='cuda:0')
# tensor([0.6924, 0.6924, 0.6924, 1.0992, 1.0992, 1.2575, 1.2575, 1.8680, 1.8680,
#         1.8680, 3.4008, 3.4008, 3.4008], device='cuda:0')
# tensor([40.0000,  0.5000,  0.5000,  0.2160], device='cuda:0')
# tensor([ 0.0000, -0.5000, -0.5000, -0.2160], device='cuda:0')        8e6

# tensor([ 1.2106e-08,  6.1583e-09,  2.4578e-08,  2.9006e-08,  2.2333e-09,
#          2.4120e-08, -1.0663e-08,  6.7053e-09, -5.5939e-10, -1.3546e-08,
#         -9.7969e-09, -3.0122e-08,  3.5178e-08])
# tensor([0.4448, 0.4448, 0.4448, 0.8265, 0.8265, 1.0639, 1.0639, 1.2651, 1.2651,
#         1.2651, 2.0087, 2.0087, 2.0087])
# tensor([40.0000,  0.5000,  0.5000,  0.2160])
# tensor([-0.0000, -0.5000, -0.5000, -0.2160])        8e6_normal


# tensor([ 5.1409e-04, -2.5742e-06,  2.9052e-02,  1.0914e-02,  1.3597e-02,
#          4.6335e-03, -6.1454e-03,  1.2510e-03, -2.9151e-03,  6.5534e-03,
#          3.2437e-02,  2.4841e-02,  2.1411e-02], device='cuda:0')
# tensor([0.5460, 0.5466, 0.5218, 0.8333, 0.8176, 1.1591, 1.1624, 1.3324, 1.3371,
#         1.3170, 2.2232, 2.2270, 2.3840], device='cuda:0')
# tensor([40.0000,  0.5000,  0.5000,  0.2160], device='cuda:0')
# tensor([ 0.0000, -0.5000, -0.5000, -0.2160], device='cuda:0')   4e6_sort_1


# tensor([ 3.0682e-05, -2.5663e-06, -2.8249e-05,  2.1724e-05, -2.3048e-05,
#         -5.3000e-05,  1.6769e-05, -4.9739e-05, -8.7115e-05, -1.9650e-04,
#         -1.5815e-06,  1.9269e-04, -2.9728e-04], device='cuda:0')
# tensor([0.4305, 0.4305, 0.4304, 0.6404, 0.6406, 0.6404, 0.6403, 1.1853, 1.1853,
#         1.1853, 1.8881, 1.8881, 1.8882], device='cuda:0')
# tensor([40.0000,  0.5000,  0.5000,  0.2160], device='cuda:0')
# tensor([-0.0000, -0.5000, -0.5000, -0.2160], device='cuda:0')        QUA_6e6_near_0

# tensor([ 2.8495e-05, -2.1259e-05, -5.1220e-05,  5.3406e-05,  1.8859e-05,
#          9.4387e-05, -2.0133e-06, -5.1457e-05,  8.0693e-06,  5.6623e-05,
#         -1.0673e-04, -3.7677e-04,  1.8596e-04], device='cuda:0')
# tensor([0.6152, 0.6153, 0.6153, 0.9512, 0.9513, 0.9512, 0.9511, 1.5845, 1.5845,
#         1.5845, 2.9678, 2.9676, 2.9676], device='cuda:0')
# tensor([40.0000,  0.5000,  0.5000,  0.2160], device='cuda:0')
# tensor([-0.0000, -0.5000, -0.5000, -0.2160], device='cuda:0')      QUA_1e7_near_2


# tensor([ 1.0589e-05,  1.1117e-05,  1.3680e-05,  1.6457e-05, -4.5605e-05,
#         -2.8571e-04, -2.4076e-04, -9.0192e-05,  3.2181e-05, -1.3122e-04,
#          3.5321e-06, -3.5471e-05,  2.1887e-05], device='cuda:0')
# tensor([0.3367, 0.3367, 0.3367, 0.7798, 0.7798, 0.8725, 0.8727, 0.9322, 0.9321,
#         0.9320, 0.2500, 0.2500, 0.2500], device='cuda:0')
# tensor([40.0000,  0.0837,  0.0982,  0.1114], device='cuda:0')
# tensor([-0.0000, -0.0837, -0.0981, -0.1113], device='cuda:0')



# tensor([ 7.6001e-05, -5.2334e-05, -9.1711e-06, -8.7536e-05, -2.3462e-05,
#          3.4021e-05,  1.2674e-04,  1.1397e-04,  1.3111e-04,  2.1381e-05,
#          3.8638e-05, -2.8105e-06,  9.7870e-06], device='cuda:0')
# tensor([0.4629, 0.4629, 0.4629, 0.8996, 0.8997, 0.9547, 0.9547, 1.0755, 1.0754,
#         1.0755, 0.2868, 0.2868, 0.2868], device='cuda:0')
# tensor([40.0000,  0.0837,  0.0982,  0.1114], device='cuda:0')
# tensor([-0.0000, -0.0837, -0.0982, -0.1114], device='cuda:0')



# class MpcBcNetwork(nn.Module):
#     def __init__(self, n_states, n_controls):
#         super(MpcBcNetwork, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(n_states, 128),
#             nn.ReLU(),
#             nn.Linear(128, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, n_controls),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         return self.model(x)

# class MpcBcNetwork(nn.Module):
#     def __init__(self, n_states, n_controls):
#         super(MpcBcNetwork,self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(n_states, 50),
#             nn.ReLU(),
#             nn.Linear(50, 100),
#             nn.ReLU(),
#             nn.Linear(100, 200),
#             nn.ReLU(),
#             nn.Linear(200, 100),
#             nn.ReLU(),
#             nn.Linear(100, n_controls),
#             nn.Sigmoid()
#         )
#
#     def forward(self,x):
#         x = self.model(x)
#         return x

# class MpcBcNetwork(nn.Module):
#     def __init__(self, n_states, n_controls):
#         super(MpcBcNetwork,self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(n_states, 100),
#             nn.ReLU(),
#             nn.Linear(100, 200),
#             nn.ReLU(),
#             nn.Linear(200, 200),
#             nn.ReLU(),
#             nn.Linear(200, 200),
#             nn.ReLU(),
#             nn.Linear(200, 200),
#             nn.ReLU(),
#             nn.Linear(200, 100),
#             nn.ReLU(),
#             nn.Linear(100, n_controls),
#             nn.Sigmoid()
#         )
#
#     def forward(self,x):
#         x = self.model(x)
#         return x

class MpcBcNetwork(nn.Module):
    def __init__(self, n_states, n_controls):
        super(MpcBcNetwork,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_states, 100),
            nn.ReLU(),
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, n_controls),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.model(x)
        return x


# class MpcBcNetwork(nn.Module):
#     def __init__(self, input_dim=13, output_dim=4):
#         super().__init__()
#
#         # self.max_vals_tensor = torch.tensor([4.17, 3.41], device='cuda:0')
#         # self.min_vals_tensor = torch.tensor([1.69, 3.0], device='cuda:0')
#
#         assert input_dim <= 16, "Only supports input_dim <= 16 for 4x4 reshape"
#
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#
#         # 卷积层（输入为1通道 4×4）
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=1),  # [B, 16, 4, 4]
#             nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1), # [B, 32, 4, 4]
#             nn.ReLU(),
#             nn.Flatten()                                 # [B, 32*4*4]
#         )
#
#         self.fc_layers = nn.Sequential(
#             nn.Linear(32*6*6, 50),
#             nn.ReLU(),
#             nn.Linear(50, 100),
#             nn.ReLU(),
#             nn.Linear(100, 200),
#             nn.ReLU(),
#             nn.Linear(200, 100),
#             nn.ReLU(),
#             nn.Linear(100, output_dim),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         """
#         输入: x ∈ [B, 12]
#         输出: y ∈ [B, output_dim]
#         """
#         B = x.shape[0]
#
#         # 补零到16维（4×4）
#         if x.shape[1] < 16:
#             pad = torch.zeros(B, 16 - x.shape[1], device=x.device)
#             x = torch.cat([x, pad], dim=1)
#
#         x = x.view(B, 1, 4, 4)  # reshape 为 [B, 1, 4, 4]
#         x = self.conv_layers(x)
#         x = self.fc_layers(x)
#         return x


# class MpcBcNetwork(nn.Module):
#     def __init__(self, n_states, n_controls):
#         super(MpcBcNetwork,self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(n_states, 100),
#             # nn.LayerNorm(200),
#             nn.ReLU(),
#             nn.Linear(100, 200),
#             # nn.LayerNorm(400),
#             nn.ReLU(),
#             nn.Linear(200, 300),
#             # nn.LayerNorm(600),
#             nn.ReLU(),
#             nn.Linear(300, 300),
#             # nn.LayerNorm(300),
#             nn.ReLU(),
#             nn.Linear(300, 100),
#             # nn.LayerNorm(100),
#             nn.ReLU(),
#             nn.Linear(100, n_controls),
#             nn.Sigmoid()
#         )
#
#     def forward(self,x):
#         # x = self.model(x)*(self.max_vals_tensor - self.min_vals_tensor) + self.min_vals_tensor
#         x = self.model(x)
#         return x


# class MpcBcNetwork(nn.Module):
#     def __init__(self, n_states, n_controls):
#         super(MpcBcNetwork,self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(n_states, 100),
#             nn.ReLU(),
#             nn.Linear(100, 200),
#             nn.ReLU(),
#             nn.Linear(200, 300),
#             nn.ReLU(),
#             nn.Linear(300, 400),
#             nn.ReLU(),
#             nn.Linear(400, 300),
#             nn.ReLU(),
#             nn.Linear(300, 100),
#             nn.ReLU(),
#             nn.Linear(100, n_controls),
#             nn.Sigmoid()
#         )
#
#     def forward(self,x):
#         # x = self.model(x)*(self.max_vals_tensor - self.min_vals_tensor) + self.min_vals_tensor
#         x = self.model(x)
#         return x


# class MpcBcNetwork(nn.Module):
#     def __init__(self, n_states, n_controls):
#         super(MpcBcNetwork, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(n_states, 128),
#             nn.ReLU(),
#             nn.Linear(128, 256),
#             nn.ReLU(),
#             nn.Linear(256, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, n_controls)
#         )
#
#     def forward(self, x):
#         return self.model(x)





# class MpcBcNetwork(nn.Module):
#     def __init__(self, n_states, n_controls):
#         super(MpcBcNetwork,self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(n_states, 512),
#             nn.LayerNorm(512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.LayerNorm(512),
#             nn.ReLU(),
#             nn.Linear(512, 600),
#             nn.LayerNorm(600),
#             nn.ReLU(),
#             nn.Linear(600, 500),
#             nn.LayerNorm(500),
#             nn.ReLU(),
#             nn.Linear(500, 400),
#             nn.LayerNorm(400),
#             nn.ReLU(),
#             nn.Linear(400, 300),
#             nn.LayerNorm(300),
#             nn.ReLU(),
#             nn.Linear(300, 200),
#             nn.LayerNorm(200),
#             nn.ReLU(),
#             nn.Linear(200, n_controls),
#             nn.Sigmoid()
#         )
#
#     def forward(self,x):
#         # x = self.model(x)*(self.max_vals_tensor - self.min_vals_tensor) + self.min_vals_tensor
#         x = self.model(x)
#         return x


# class MpcBcNetwork(nn.Module):
#     def __init__(self, input_dim=13, output_dim=4):
#         super().__init__()
#
#         assert input_dim <= 16, "Only supports input_dim <= 16 for 4x4 reshape"
#
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#
#         # 卷积层（输入为1通道 4×4）
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=1),  # [B, 16, 4, 4],3,1,1
#             nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1), # [B, 32, 4, 4]
#             nn.ReLU(),
#             nn.Flatten()                                 # [B, 32*4*4]
#         )
#
#         self.fc_layers = nn.Sequential(
#             nn.Linear(32*6*6, 100),
#             nn.LayerNorm(100),
#             nn.ReLU(),
#             nn.Linear(100, 80),
#             nn.LayerNorm(80),
#             nn.ReLU(),
#             # nn.Dropout(0.3),
#             nn.Linear(80, 60),
#             nn.LayerNorm(60),
#             nn.ReLU(),
#             # nn.Dropout(0.3),
#             nn.Linear(60, 30),
#             nn.LayerNorm(30),
#             nn.ReLU(),
#             nn.Linear(30, output_dim),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         """
#         输入: x ∈ [B, 12]
#         输出: y ∈ [B, output_dim]
#         """
#         B = x.shape[0]
#
#         # 补零到16维（4×4）
#         if x.shape[1] < 16:
#             pad = torch.zeros(B, 16 - x.shape[1], device=x.device)
#             x = torch.cat([x, pad], dim=1)
#
#         x = x.view(B, 1, 4, 4)  # reshape 为 [B, 1, 4, 4]
#         x = self.conv_layers(x)
#         x = self.fc_layers(x)
#         # x = self.fc_layers(x)*(self.max_vals_tensor - self.min_vals_tensor) + self.min_vals_tensor
#         return x







    # class MpcBcNetwork(nn.Module):
    #     def __init__(self, input_dim=13, output_dim=4):
    #         super().__init__()
    #
    #         self.max_vals_tensor = torch.tensor([4.17, 3.41], device='cuda:0')
    #         self.min_vals_tensor = torch.tensor([1.69, 3.0], device='cuda:0')
    #
    #         assert input_dim <= 16, "Only supports input_dim <= 16 for 4x4 reshape"
    #
    #         self.input_dim = input_dim
    #         self.output_dim = output_dim
    #
    #         # 卷积层（输入为1通道 4×4）
    #         self.conv_layers = nn.Sequential(
    #             nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=1),  # [B, 16, 4, 4]
    #             nn.ReLU(),
    #             nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1), # [B, 32, 4, 4]
    #             nn.ReLU(),
    #             nn.Flatten()                                 # [B, 32*4*4]
    #         )
    #
    #         self.fc_layers = nn.Sequential(
    #             nn.Linear(32*6*6, 600),
    #             nn.LayerNorm(600),
    #             nn.ReLU(),
    #             nn.Linear(600, 500),
    #             nn.LayerNorm(500),
    #             nn.ReLU(),
    #             nn.Linear(500, 400),
    #             nn.LayerNorm(400),
    #             nn.ReLU(),
    #             nn.Linear(400, 300),
    #             nn.LayerNorm(300),
    #             nn.ReLU(),
    #             nn.Linear(300, 200),
    #             nn.LayerNorm(200),
    #             nn.ReLU(),
    #             nn.Linear(200, output_dim),
    #             nn.Sigmoid()
    #         )
    #
    #     def forward(self, x):
    #         """
    #         输入: x ∈ [B, 12]
    #         输出: y ∈ [B, output_dim]
    #         """
    #         B = x.shape[0]
    #
    #         # 补零到16维（4×4）
    #         if x.shape[1] < 16:
    #             pad = torch.zeros(B, 16 - x.shape[1], device=x.device)
    #             x = torch.cat([x, pad], dim=1)
    #
    #         x = x.view(B, 1, 4, 4)  # reshape 为 [B, 1, 4, 4]
    #         x = self.conv_layers(x)
    #         x = self.fc_layers(x)
    #         return x





# class MpcBcNetwork(nn.Module):
#     def __init__(self, input_dim=14, output_dim=4):
#         super().__init__()
#
#         assert input_dim <= 16, "Only supports input_dim <= 16 for 4x4 reshape"
#
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#
#         # 卷积层（输入为1通道 4×4）
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=1),  # [B, 16, 4, 4]
#             nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1), # [B, 32, 4, 4]
#             nn.ReLU(),
#             nn.Flatten()                                 # [B, 32*4*4]
#         )
#
#         self.fc_layers = nn.Sequential(
#             nn.Linear(32*6*6, 1600),
#             nn.LayerNorm(1600),
#             nn.ReLU(),
#             nn.Linear(1600, 1200),
#             nn.LayerNorm(1200),
#             nn.ReLU(),
#             nn.Linear(1200, 800),
#             nn.LayerNorm(800),
#             nn.ReLU(),
#             nn.Linear(800, 600),
#             nn.LayerNorm(600),
#             nn.ReLU(),
#             nn.Linear(600, 400),
#             nn.LayerNorm(400),
#             nn.ReLU(),
#             nn.Linear(400, 200),
#             nn.LayerNorm(200),
#             nn.ReLU(),
#             nn.Linear(200, output_dim),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         """
#         输入: x ∈ [B, 12]
#         输出: y ∈ [B, output_dim]
#         """
#         B = x.shape[0]
#
#         # 补零到16维（4×4）
#         if x.shape[1] < 16:
#             pad = torch.zeros(B, 16 - x.shape[1], device=x.device)
#             x = torch.cat([x, pad], dim=1)
#
#         x = x.view(B, 1, 4, 4)  # reshape 为 [B, 1, 4, 4]
#         x = self.conv_layers(x)
#         x = self.fc_layers(x)
#         return x




# class MpcBcNetwork(nn.Module):
#     def __init__(self, input_dim=14, output_dim=4):
#         super().__init__()
#
#         assert input_dim <= 16, "Only supports input_dim <= 16 for 4x4 reshape"
#
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#
#         # 卷积层（输入为1通道 4×4）
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # [B, 16, 4, 4]
#             nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # [B, 32, 4, 4]
#             nn.ReLU(),
#             nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),  # [B, 16, 4, 4]
#             nn.ReLU(),
#             nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),  # [B, 32, 4, 4]
#             nn.ReLU(),
#             nn.Conv2d(64, 80, kernel_size=3, stride=1, padding=1),  # [B, 32, 4, 4]
#             nn.ReLU(),
#             nn.Conv2d(80, 96, kernel_size=3, stride=1, padding=1),  # [B, 32, 4, 4]
#             nn.ReLU(),
#             nn.Flatten()                                 # [B, 32*4*4]
#         )
#
#         self.fc_layers = nn.Sequential(
#             nn.Linear(96*4*4, 200),
#             nn.LayerNorm(200),
#             nn.ReLU(),
#             nn.Linear(200, 150),
#             nn.LayerNorm(150),
#             nn.ReLU(),
#             nn.Linear(150, output_dim),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         """
#         输入: x ∈ [B, 12]
#         输出: y ∈ [B, output_dim]
#         """
#         B = x.shape[0]
#
#         # 补零到16维（4×4）
#         if x.shape[1] < 16:
#             pad = torch.zeros(B, 16 - x.shape[1], device=x.device)
#             x = torch.cat([x, pad], dim=1)
#
#         x = x.view(B, 1, 4, 4)  # reshape 为 [B, 1, 4, 4]
#         x = self.conv_layers(x)
#         x = self.fc_layers(x)
#         return x

