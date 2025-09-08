import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights
import copy  
from collections import OrderedDict
import torch.nn.functional as F 
from torchvision.models import vgg16, VGG16_Weights

from transformers import BertModel, BertTokenizer, BertConfig




def sequential_slice_keep_names(module_list: nn.ModuleList,
                                start: int,
                                end: int | None = None) -> nn.Sequential:
    """
    nn.Sequential whose children are keyed by their original indices.
    """
    items = list(module_list._modules.items())[start:end]   # keeps (idx, child)
    return nn.Sequential(OrderedDict(
        (str(idx), copy.deepcopy(child)) for idx, child in items
    ))

class VGGBase(nn.Module):
    """
    Wrapped torchvision VGG-16 so our split helpers work the same
    way as for CNNBase / ResNetBase.
    """
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.features = backbone.features             # 13 conv layers
        self.avgpool  = backbone.avgpool              # AdaptiveAvgPool2d
        # keep original 4096->4096->1000 classifier but resize last layer
        self.classifier = nn.Sequential(
            *backbone.classifier[:-1],                # FC1, ReLU, Drop, FC2, ReLU, Drop
            nn.Linear(4096, num_classes)
        )
        self.fc = self.classifier[-1]                 # alias for merge

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).flatten(1)
        return self.classifier(x)



class VGGClient(nn.Module):
    def __init__(self, base, arc_config, device="cpu"):
        super().__init__()
        bb = [0, 5, 10, 17, 24, 31]            # conv-block boundaries
        self.features = sequential_slice_keep_names(
            base.features, bb[0], bb[arc_config]
        )
        self.to(device)
    def forward(self, x): return self.features(x)


class VGGServer(nn.Module):
    def __init__(self, base, arc_config, device="cpu"):
        super().__init__()
        bb = [0, 5, 10, 17, 24, 31]
        self.features = sequential_slice_keep_names(
            base.features, bb[arc_config], bb[-1]
        )
        self.avgpool    = base.avgpool
        self.classifier = copy.deepcopy(base.classifier)
        self.fc = self.classifier[-1]           # alias
        self.to(device)
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).flatten(1)
        return self.classifier(x)

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, groups=8, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.gn1   = nn.GroupNorm(groups, out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.gn2   = nn.GroupNorm(groups, out_c)

        self.skip  = (
            nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.GroupNorm(groups, out_c)
            )
            if (in_c != out_c or downsample) else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.relu(out + identity)


class CNNBase(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super().__init__()

        # ── renamed stem ───────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True)
        )

        self.init = self.stem      # ← ❶ alias for backward compatibility
        # ------------------------------------------------

        self.blocks = nn.ModuleList([
            ConvBlock( 64,  64, downsample=False),
            ConvBlock( 64, 128, downsample=True ),
            ConvBlock(128, 256, downsample=True ),
            ConvBlock(256, 512, downsample=True ),
        ])

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.drop    = nn.Dropout(p=0.5)
        self.fc1     = nn.Linear(512, 1024)
        self.fc2     = nn.Linear(1024, num_classes)
        self.fc = self.fc2

    def forward(self, x):
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.avgpool(x).flatten(1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)

class CNNClient(nn.Module):
    def __init__(self, base_model, arc_config, device='cpu'):
        super().__init__()
        self.init = copy.deepcopy(base_model.init) 
        self.blocks = sequential_slice_keep_names(
            base_model.blocks, 0, arc_config            
        )
        self.to(device)

    def forward(self, x):
        x = self.init(x)
        x = self.blocks(x)
        return x

class CNNServer(nn.Module):
    def __init__(self, base_model, arc_config, device="cpu"):
        super().__init__()

        self.blocks   = sequential_slice_keep_names(
            base_model.blocks, arc_config, None
        )
        self.avgpool  = base_model.avgpool            # share OK

        # copy **both** FC layers and dropout
        self.fc1  = copy.deepcopy(base_model.fc1)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.5)
        self.fc2  = copy.deepcopy(base_model.fc2)

        # alias so merge-back still works
        self.fc = self.fc2

        self.to(device)

    def forward(self, x):
        x = self.blocks(x)
        x = self.avgpool(x).flatten(1)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

        

# ── “template” full model (rarely trained directly) ────────────────────
class ResNetBase(nn.Module):
    def __init__(self, init, blocks, avgpool, feature_dim, num_classes):
        super().__init__()
        self.init = init          # already a Sequential
        self.blocks = blocks      # ModuleList with 4 residual groups
        self.avgpool = avgpool
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.init(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.fc(x)

# ── client half ────────────────────────────────────────────────────────
class ResNetClient(nn.Module):
    def __init__(self, full: ResNetBase, arc_config: int, device="cpu"):
        super().__init__()
        self.init   = copy.deepcopy(full.init)
        self.blocks = sequential_slice_keep_names(full.blocks, 0, arc_config)
        self.to(device)

    def forward(self, x):
        return self.blocks(self.init(x))

# ── server half ────────────────────────────────────────────────────────
class ResNetServer(nn.Module):
    def __init__(self, full: ResNetBase, arc_config: int, device="cpu"):
        super().__init__()
        self.blocks   = sequential_slice_keep_names(full.blocks, arc_config, None)
        self.avgpool  = copy.deepcopy(full.avgpool)
        self.fc = copy.deepcopy(full.fc)   # ← same trick as CNN
        self.to(device)

    def forward(self, x):
        x = self.blocks(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

class BERTBase(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.fc = self.classifier  # alias

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)

class BERTClient(nn.Module):
    def __init__(self, base_model, arc_config, device="cpu"):
        super().__init__()
        assert arc_config in [0, 6, 9, 12], "arc_config must be 0, 6, 9 or 12 for BERT"
        self.embeddings = copy.deepcopy(base_model.bert.embeddings)
        self.encoder_layers = nn.ModuleList(
            [copy.deepcopy(layer) for layer in base_model.bert.encoder.layer[:arc_config]]
        )
        self.to(device)

    def get_extended_attention_mask(self, attention_mask: torch.Tensor, input_shape: torch.Size) -> torch.Tensor:
        """
        Makes the 2D attention mask broadcastable to [batch_size, num_heads, seq_length, seq_length].
        This is a standard Hugging Face utility function.
        """
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=self.embeddings.word_embeddings.weight.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, input_ids, attention_mask):
        # Create the correct 4D attention mask for the encoder
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_ids.shape)
        
        hidden_states = self.embeddings(input_ids)
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask=extended_attention_mask)[0]
            
        # The client only needs to return the final hidden states
        return hidden_states

class BERTServer(nn.Module):
    def __init__(self, base_model, arc_config, device="cpu"):
        super().__init__()
        assert arc_config in [0, 6, 9, 12], "arc_config must be 0, 6, 9 or 12 for BERT"
        self.encoder_layers = nn.ModuleList(
            [copy.deepcopy(layer) for layer in base_model.bert.encoder.layer[arc_config:]]
        )
        self.pooler = copy.deepcopy(base_model.bert.pooler)
        self.classifier = copy.deepcopy(base_model.classifier)
        self.fc = self.classifier
        self.to(device)

    def forward(self, hidden_states, attention_mask):
        # The server also needs to create the extended mask for its layers.
        # It gets the original 2D mask from the trainer loop.
        extended_attention_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask=extended_attention_mask)[0]
            
        pooled_output = self.pooler(hidden_states)
        logits = self.classifier(pooled_output)
        return logits


class CharLSTM(nn.Module):
    """A simple character-level LSTM model."""
    def __init__(self, num_classes, embedding_dim=256, hidden_dim=512, n_layers=2):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    # =================== START OF THE FIX ===================
    # Update the forward method signature to accept keyword arguments
    def forward(self, input_ids, attention_mask=None, hidden=None):
        # We now use 'input_ids' as the input variable name.
        # The attention_mask is accepted but ignored, as the LSTM doesn't use it.
        # input_ids shape: [batch_size, seq_length]
        
        embedded = self.embedding(input_ids)
        # embedded shape: [batch_size, seq_length, embedding_dim]
        
        lstm_out, hidden = self.lstm(embedded, hidden)
        # lstm_out shape: [batch_size, seq_length, hidden_dim]

        # This part of the logic was already correct
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        logits = self.fc(lstm_out)
        
        # We only need to return the logits for the loss function
        # The hidden state is for stateful inference, which we aren't doing here
        return logits

# --- Splitfed versions ---

class CharLSTMClient(nn.Module):
    """Client part of the CharLSTM: embedding layer."""
    def __init__(self, base_model, device="cpu"):
        super().__init__()
        self.embedding = copy.deepcopy(base_model.embedding)
        self.to(device)

    def forward(self, input_ids, attention_mask=None):
        # We use 'input_ids' and can safely ignore the 'attention_mask'
        # as the embedding layer doesn't need it.
        return self.embedding(input_ids)

class CharLSTMServer(nn.Module):
    """Server part of the CharLSTM: LSTM layers and the final classifier."""
    def __init__(self, base_model, device="cpu"):
        super().__init__()
        self.lstm = copy.deepcopy(base_model.lstm)
        self.fc = copy.deepcopy(base_model.fc)
        self.hidden_dim = base_model.hidden_dim
        self.to(device)
        
    def forward(self, x, attention_mask=None):
        # The LSTM part does not use the attention mask, so we can ignore it.
        # x is the output from the client (embedded features)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        logits = self.fc(lstm_out)
        return logits


def create_split_vgg(num_classes: int,
                     arc_config: int,
                     base_model: nn.Module | None,
                     device: str = "cpu"):
    """
    Split a VGG-16 at conv-block boundary.
    arc_config = 1…4  (client blocks) ; server gets the rest.
    """
    if not 1 <= arc_config <= 4:
        raise ValueError("arc_config must be 1..4 for VGG-16")

    # 1. backbone -----------------------------------------------------------
    if base_model is None:
        backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    elif isinstance(base_model, VGGBase):
        backbone = copy.deepcopy(base_model)          # already wrapped
    else:                                             # raw torchvision VGG
        backbone = copy.deepcopy(base_model)
    backbone = backbone.to(device)

    # 2. wrap ---------------------------------------------------------------
    full_model = (backbone if isinstance(backbone, VGGBase)
                  else VGGBase(backbone, num_classes)).to(device)

    # 3. split --------------------------------------------------------------
    client = VGGClient(full_model, arc_config, device)
    server = VGGServer(full_model, arc_config, device)
    total_layers = 5                                  # conv blocks
    return client, server, full_model.cpu(), total_layers

def create_split_cnn(num_classes: int, arc_config: int, device='cpu', in_channels=3):
    # Create base model
    base_model = CNNBase(num_classes, in_channels=in_channels)
    total_layers = get_total_layers(base_model)

    
    # Validate arc_config
    if not (0 <= arc_config <= total_layers):
        raise ValueError(f"Invalid arc_config={arc_config}, must be 0-{total_layers}")

    # Create client and server models
    client_model = CNNClient(base_model, arc_config, device=device)
    server_model = CNNServer(base_model, arc_config, device=device)

    ###-----DEBUG: Checking the clinet and server model split correctly (ok)----###
    # print("\n[INFO] Client model:\n", client_model)
    # print("\n[INFO] Server model:\n", server_model)
    # print("\n[INFO] Full model:\n", base_model)
    # print(f"\n[INFO] Total blocks in CNNBase: {total_layers}")
    
    return client_model, server_model, base_model, total_layers

def create_split_resnet18(num_classes: int,
                          arc_config: int,
                          base_model: nn.Module,
                          device: str = "cpu"):
    """
    Split a ResNet-18 that is either
      • a `ResNetBase` we created earlier, or
      • a raw torchvision `resnet18`.

    Returns: client_model, server_model, full_model, total_layers
    """
    # ------------------------------------------------------------------ case 1
    if isinstance(base_model, ResNetBase):
        full = copy.deepcopy(base_model).to(device)

    # ------------------------------------------------------------------ case 2
    elif hasattr(base_model, "conv1"):           # any torchvision ResNet-18
        tv  = copy.deepcopy(base_model).to(device)  # or load a fresh one
        init = nn.Sequential(tv.conv1, tv.bn1, tv.relu, tv.maxpool)
        blocks = nn.ModuleList([tv.layer1, tv.layer2, tv.layer3, tv.layer4])
        full = ResNetBase(init, blocks, tv.avgpool,
                          tv.fc.in_features, num_classes).to(device)
        # resize FC if class count differs
        if full.fc.out_features != num_classes:
            full.fc = nn.Linear(tv.fc.in_features, num_classes)

    # ------------------------------------------------------------------ no model yet
    else:
        raise ValueError("base_model must be ResNetBase or torchvision ResNet-18")

    # ------------- split ----------------------------------------------
    total_layers = len(full.blocks)          # = 4
    if not 1 <= arc_config <= total_layers - 1:
        raise ValueError(f"arc_config must be 1–{total_layers-1}")

    client = ResNetClient(full, arc_config, device)
    server = ResNetServer(full, arc_config, device)

    ###-----DEBUG: Checking the clinet and server model split correctly (ok)----###
    # print("\n[INFO] Client model:\n", client)
    # print("\n[INFO] Server model:\n", server)
    # print("\n[INFO] Full model:\n", full)
    # print(f"\n[INFO] Total blocks in ResnetBase: {total_layers}")

    return client, server, full.cpu(), total_layers

def create_split_bert(num_classes, arc_config, base_model=None, device="cpu"):
    if base_model is None:
        base_model = BERTBase(num_classes)

    total_layers = 12
    if arc_config not in [0, 6, 9, 12]:
        raise ValueError("arc_config must be one of [0, 6, 9, 12] for BERT")

    client = BERTClient(base_model, arc_config, device)
    server = BERTServer(base_model, arc_config, device)
    return client, server, base_model.cpu(), total_layers

def create_split_charlstm(num_classes, arc_config, base_model=None, device="cpu"):
    """
    Builds and splits a CharLSTM model.
    The split for LSTM is simple: client has embeddings, server has the rest.
    The arc_config is unused here but kept for compatibility.
    """
    if base_model is None:
        base_model = CharLSTM(num_classes).to(device)

    client_model = CharLSTMClient(base_model, device)
    server_model = CharLSTMServer(base_model, device)
    
    # For LSTM, we can consider the "total layers" to be the LSTM layers + classifier
    total_layers = base_model.n_layers + 1 
    
    return client_model, server_model, base_model.cpu(), total_layers



def create_split_model(model_name: str, num_classes: int, arc_config: int, base_model=None, device='cpu', in_channels=3):
    """
    Create split models for different architectures.
    """
    if model_name.lower() == 'resnet18':
        return create_split_resnet18(num_classes, arc_config, base_model, device)
    elif model_name.lower() == 'cnn':
        return create_split_cnn(num_classes, arc_config, device, in_channels)
    elif model_name.lower() == "vgg16":
        return create_split_vgg(num_classes, arc_config, base_model, device)
    elif model_name.lower() == "bert":
        return create_split_bert(num_classes, arc_config, base_model, device)
    elif model_name.lower() == 'charlstm':
        return create_split_charlstm(num_classes, arc_config, base_model, device)

    else:
        raise NotImplementedError(f"Model {model_name} not supported.")

def get_total_layers(model: nn.Module) -> int:
    if hasattr(model, 'blocks') and isinstance(model.blocks, (nn.Sequential, nn.ModuleList)):
        return len(model.blocks)
    elif hasattr(model, 'bert') and hasattr(model.bert, 'encoder'):
        return len(model.bert.encoder.layer)  # For BERT
    elif hasattr(model, 'lstm') and isinstance(model.lstm, nn.LSTM):
        return model.lstm.num_layers
    raise NotImplementedError("Unsupported model architecture.")


def create_global_model(model_name: str,
                        num_classes: int,
                        in_channels: int,
                        device: str):
    """
    Return a *full* network (no split), ready to be sliced later.
    """
    if model_name.lower() == "cnn":
        return CNNBase(num_classes, in_channels).to(device)

    elif model_name.lower() == "resnet18":
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        init = nn.Sequential(backbone.conv1, backbone.bn1,
                             backbone.relu, backbone.maxpool)
        blocks = nn.ModuleList([backbone.layer1, backbone.layer2,
                                backbone.layer3, backbone.layer4])
        feature_dim = backbone.fc.in_features
        full = ResNetBase(init, blocks, backbone.avgpool,
                          feature_dim, num_classes).to(device)

        # Resize the FC layer if you’re *not* using 1000-class ImageNet
        if num_classes != backbone.fc.out_features:
            full.fc = nn.Linear(feature_dim, num_classes)
        return full

    # ───────────────────────── 3. VGG-16 (ImageNet pre-trained) ────────────
    elif model_name == "vgg16":
        tv = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        full = VGGBase(tv, num_classes).to(device)     # VGGBase defined earlier
        return full

    elif model_name.lower() == "bert":
        return BERTBase(num_classes).to(device)
    
    elif model_name.lower() == 'charlstm':
        return CharLSTM(num_classes).to(device)

    else:
        raise NotImplementedError(f"{model_name} not supported")