"""
MMoE (Multi-gate Mixture-of-Experts) Module for Multi-task Learning
Adapted for medical image classification tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """
    Single Expert Network
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super(Expert, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.ReLU(inplace=True))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)


class GatingNetwork(nn.Module):
    """
    Gating Network to compute expert weights for each task
    """
    def __init__(self, input_dim, hidden_dims, num_experts, dropout=0.1):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            num_experts: Number of experts
            dropout: Dropout rate
        """
        super(GatingNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_experts))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Returns:
            gate_weights: [batch_size, num_experts] - softmax weights
        """
        logits = self.network(x)
        gate_weights = F.softmax(logits, dim=1)
        return gate_weights


class MMoELayer(nn.Module):
    """
    MMoE (Multi-gate Mixture-of-Experts) Layer
    """
    def __init__(self, 
                 input_dim, 
                 num_experts=3,
                 expert_hidden_dims=[256, 128],
                 expert_output_dim=64,
                 gate_hidden_dims=[64, 32],
                 num_tasks=2,
                 dropout=0.1):
        """
        Args:
            input_dim: Input feature dimension
            num_experts: Number of expert networks
            expert_hidden_dims: Hidden dimensions for each expert
            expert_output_dim: Output dimension of each expert
            gate_hidden_dims: Hidden dimensions for gating networks
            num_tasks: Number of tasks
            dropout: Dropout rate
        """
        super(MMoELayer, self).__init__()
        
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.expert_output_dim = expert_output_dim
        
        # Create multiple expert networks
        self.experts = nn.ModuleList([
            Expert(input_dim, expert_hidden_dims, expert_output_dim, dropout)
            for _ in range(num_experts)
        ])
        
        # Create gating network for each task
        self.gates = nn.ModuleList([
            GatingNetwork(input_dim, gate_hidden_dims, num_experts, dropout)
            for _ in range(num_tasks)
        ])
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, input_dim]
        
        Returns:
            task_inputs: List of tensors, one for each task [batch_size, expert_output_dim]
        """
        batch_size = x.size(0)
        
        # Compute expert outputs: [num_experts, batch_size, expert_output_dim]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=0)
        
        # Compute task-specific weighted combinations
        task_inputs = []
        for task_idx in range(self.num_tasks):
            # Get gate weights for this task: [batch_size, num_experts]
            gate_weights = self.gates[task_idx](x)
            
            # Weighted sum of expert outputs: [batch_size, expert_output_dim]
            # expert_outputs: [num_experts, batch_size, expert_output_dim]
            # gate_weights: [batch_size, num_experts]
            # We need to transpose and multiply
            weighted_output = torch.einsum('ebd,be->bd', expert_outputs, gate_weights)
            
            task_inputs.append(weighted_output)
        
        return task_inputs


class MMoEModel(nn.Module):
    """
    Complete MMoE Model with task-specific towers
    """
    def __init__(self,
                 input_dim,
                 num_experts=3,
                 expert_hidden_dims=[256, 128],
                 expert_output_dim=64,
                 gate_hidden_dims=[64, 32],
                 tower_hidden_dims=[128, 64],
                 output_dims=[2, 6],  # [benign_malignant_classes, birads_classes]
                 dropout=0.1):
        """
        Args:
            input_dim: Input feature dimension
            num_experts: Number of expert networks
            expert_hidden_dims: Hidden dimensions for experts
            expert_output_dim: Output dimension of experts
            gate_hidden_dims: Hidden dimensions for gates
            tower_hidden_dims: Hidden dimensions for task towers
            output_dims: List of output dimensions for each task
            dropout: Dropout rate
        """
        super(MMoEModel, self).__init__()
        
        self.num_tasks = len(output_dims)
        
        # MMoE layer
        self.mmoe = MMoELayer(
            input_dim=input_dim,
            num_experts=num_experts,
            expert_hidden_dims=expert_hidden_dims,
            expert_output_dim=expert_output_dim,
            gate_hidden_dims=gate_hidden_dims,
            num_tasks=self.num_tasks,
            dropout=dropout
        )
        
        # Task-specific towers
        self.towers = nn.ModuleList()
        for task_idx, output_dim in enumerate(output_dims):
            tower_layers = []
            prev_dim = expert_output_dim
            
            for hidden_dim in tower_hidden_dims:
                tower_layers.append(nn.Linear(prev_dim, hidden_dim))
                tower_layers.append(nn.ReLU(inplace=True))
                tower_layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim
            
            tower_layers.append(nn.Linear(prev_dim, output_dim))
            
            self.towers.append(nn.Sequential(*tower_layers))
        
        # Initialize tower weights
        self._init_weights()
    
    def _init_weights(self):
        for tower in self.towers:
            for m in tower.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, input_dim]
        
        Returns:
            outputs: List of output tensors for each task
        """
        # Get task-specific inputs from MMoE
        task_inputs = self.mmoe(x)
        
        # Pass through task-specific towers
        outputs = [self.towers[i](task_inputs[i]) for i in range(self.num_tasks)]
        
        return outputs


class SimpleMMoE(nn.Module):
    """
    Simplified MMoE for quick integration (similar to your example)
    """
    def __init__(self,
                 input_dim,
                 num_experts=4,
                 expert_dims=[64, 32, 16],
                 gate_dims=[16, 8],
                 task_dims=[32, 16],
                 output_task1_dim=1,
                 output_task2_dim=1):
        super(SimpleMMoE, self).__init__()
        
        self.num_experts = num_experts
        self.expert_output_dim = expert_dims[-1]
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dims[0]),
                nn.ReLU(),
                nn.Linear(expert_dims[0], expert_dims[1]),
                nn.ReLU(),
                nn.Linear(expert_dims[1], expert_dims[2]),
                nn.ReLU()
            ) for _ in range(num_experts)
        ])
        
        # Task 1 tower
        self.task1_head = nn.Sequential(
            nn.Linear(expert_dims[2], task_dims[0]),
            nn.ReLU(),
            nn.Linear(task_dims[0], task_dims[1]),
            nn.ReLU(),
            nn.Linear(task_dims[1], output_task1_dim),
            nn.Sigmoid()
        )
        
        # Task 2 tower
        self.task2_head = nn.Sequential(
            nn.Linear(expert_dims[2], task_dims[0]),
            nn.ReLU(),
            nn.Linear(task_dims[0], task_dims[1]),
            nn.ReLU(),
            nn.Linear(task_dims[1], output_task2_dim),
            nn.Sigmoid()
        )
        
        # Gating networks
        self.gating1_network = nn.Sequential(
            nn.Linear(input_dim, gate_dims[0]),
            nn.ReLU(),
            nn.Linear(gate_dims[0], gate_dims[1]),
            nn.ReLU(),
            nn.Linear(gate_dims[1], num_experts),
            nn.Softmax(dim=1)
        )
        
        self.gating2_network = nn.Sequential(
            nn.Linear(input_dim, gate_dims[0]),
            nn.ReLU(),
            nn.Linear(gate_dims[0], gate_dims[1]),
            nn.ReLU(),
            nn.Linear(gate_dims[1], num_experts),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # Compute gate weights
        gates1 = self.gating1_network(x)
        gates2 = self.gating2_network(x)
        
        batch_size = x.size(0)
        device = x.device
        
        task1_inputs = torch.zeros(batch_size, self.expert_output_dim).to(device)
        task2_inputs = torch.zeros(batch_size, self.expert_output_dim).to(device)
        
        # Weighted sum of expert outputs
        for i in range(self.num_experts):
            expert_output = self.experts[i](x)
            task1_inputs += expert_output * gates1[:, i].unsqueeze(1)
            task2_inputs += expert_output * gates2[:, i].unsqueeze(1)
        
        # Task-specific outputs
        task1_outputs = self.task1_head(task1_inputs)
        task2_outputs = self.task2_head(task2_inputs)
        
        return task1_outputs, task2_outputs


if __name__ == '__main__':
    print("="*60)
    print("Testing MMoE Models")
    print("="*60)
    
    # Test configuration
    batch_size = 16
    input_dim = 640  # Feature dimension from MambaVision backbone
    
    # Test 1: Full MMoE Model
    print("\n1. Testing Full MMoE Model")
    print("-"*60)
    model1 = MMoEModel(
        input_dim=input_dim,
        num_experts=4,
        expert_hidden_dims=[256, 128],
        expert_output_dim=64,
        gate_hidden_dims=[64, 32],
        tower_hidden_dims=[128, 64],
        output_dims=[2, 6],  # 2 classes for benign/malignant, 6 for BI-RADS
        dropout=0.1
    )
    
    dummy_input = torch.randn(batch_size, input_dim)
    outputs = model1(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Task 1 output shape: {outputs[0].shape}")
    print(f"Task 2 output shape: {outputs[1].shape}")
    
    total_params = sum(p.numel() for p in model1.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test 2: Simple MMoE Model
    print("\n2. Testing Simple MMoE Model")
    print("-"*60)
    model2 = SimpleMMoE(
        input_dim=input_dim,
        num_experts=4,
        expert_dims=[64, 32, 16],
        gate_dims=[16, 8],
        task_dims=[32, 16],
        output_task1_dim=2,
        output_task2_dim=6
    )
    
    task1_out, task2_out = model2(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Task 1 output shape: {task1_out.shape}")
    print(f"Task 2 output shape: {task2_out.shape}")
    
    total_params = sum(p.numel() for p in model2.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
