import torchaudio
import torchaudio.transforms as T
import torch.nn as nn

class SpeechEmotionCNN(nn.Module):
    def __init__(self):
        super(SpeechEmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.fc1 = nn.Linear(32 * 20 * 20, 128)
        self.fc2 = nn.Linear(128, 7)  # Assuming 7 emotion classes

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x

def process_speech(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    transform = T.MelSpectrogram()
    mel_spec = transform(waveform)
    model = SpeechEmotionCNN()
    with torch.no_grad():
        logits = model(mel_spec.unsqueeze(0))  # Add batch dimension
    return torch.softmax(logits, dim=-1).detach().numpy()
