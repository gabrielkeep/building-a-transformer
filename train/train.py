import torch 
import torch.nn as nn
import torch.optim as optim
from src.model import Transformer
from tokenizador.tokenizer import Tokenizer


EPOCHS = 100
LEARNING_RATE = 1e-4



tkn = Tokenizer("""
The functioning of biological neurons is a cornerstone of neuroscience and serves as the conceptual blueprint for the artificial neural networks you are currently building. 
To provide you with a robust dataset for training your Transformer—especially one capable of capturing complex dependencies—I have prepared a technical discourse on neuronal 
physiology.
The Biological Architecture of Information Processing
The human brain is a complex web of approximately 86 billion neurons, which are specialized cells designed to transmit information through electrical and chemical signals.
Each neuron acts as a discrete processing unit, fundamentally composed of three primary structures: the dendrites, the soma (cell body), and the axon. Information flow within
a neuron is strictly directional, following a path from the receiving dendrites to the transmitting axon terminals.

1. Signal Reception and Integration
The process begins at the dendrites, tree-like extensions that reach out to other neurons. These structures contain receptors that bind to chemicals called neurotransmitters.
When a neurotransmitter binds to a receptor, it causes a local change in the electrical potential of the cell membrane. These changes can be either excitatory (increasing the
likelihood of a signal) or inhibitory (decreasing it). The soma acts as an integrator, summing all the incoming signals. This summation is remarkably similar to the weighted 
sums performed in artificial neurons, where the cell "decides" whether the input has reached a specific threshold.

2. The Action Potential: All-or-None Transmission
If the cumulative excitation at the soma reaches a critical threshold (typically around -55mV), the neuron triggers an action potential. This is a rapid, temporary reversal of
the electrical membrane potential that travels down the axon. The action potential operates on an "all-or-none" principle; it does not vary in strength based on the stimulus
intensity. Instead, intensity is communicated through the frequency of firing—a concept known as rate coding. The axon is often insulated by a fatty layer called the myelin
sheath, which significantly accelerates signal conduction through a process known as saltatory conduction.

3. Synaptic Transmission
When the action potential reaches the axon terminals, it triggers the release of neurotransmitters into the synapse, the microscopic gap between neurons.
These chemicals diffuse across the gap and bind to the dendrites of the post-synaptic neuron, continuing the chain of communication. This synaptic connection is not static;
it is subject to neuroplasticity, where the strength of the connection changes based on activity. This biological "weight adjustment" is the fundamental mechanism of learning
and memory in the brain.

4. Neural Networks and Emergent Complexity
A single neuron's computational power is limited, but when organized into vast, hierarchical networks, emergent properties arise. These networks utilize parallel processing
and feedback loops to perform tasks ranging from basic sensory perception to high-level abstract reasoning. For a Data Science professional, understanding these biological
constraints provides deep insight into why certain architectures, like the Transformer’s attention mechanism, are so effective at modeling long-range dependencies
that mimic the associative nature of human thought.
""")

input_ids, vocab_size = tkn.forward()

# Modelo

model = Transformer(
    max_seq_len = 512,
    vocab_size=vocab_size,
    embeding_dim=512,
    n_layers=2,
    d_model=512,
    n_heads=2
)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    
    # Forward pass
    output = model(input_ids)
    
    # Compute loss
    loss = criterion(output.view(-1, vocab_size), input_ids.view(-1))
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Época {epoch} | Erro: {loss.item():.4f}")
    
torch.save(model.state_dict(), 'transformer_model.pth')