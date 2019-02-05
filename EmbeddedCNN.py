import sys, os, platform

try:
    import PyQt5
    import numpy as np
    import torch
    import torchvision

    from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton
    from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
    from PyQt5.QtGui import QPainter, QPen
    from torch.autograd import Variable
    from PIL import Image
    from torch import nn

    from acnn_packages.ACNN import Ui_MainWindow as acnn

except ImportError:
    _get_python_version = sys.version_info
    _get_platform = platform.system()
    _python_version = _get_python_version[1]
    _dependencies = ['PyQt5',
                     'Pillow',
                     'numpy',
                     'matplotlib']

    assert _get_python_version >= (3, 5)

    _with_cuda = {'linux': {'8': 'https://download.pytorch.org/whl/cu80/torch-1.0.0-cp3{0]-cp3{0}m-linux_x86_64.whl',
                            '9': 'torch',
                            '10': 'https://download.pytorch.org/whl/cu100/torch-1.0.0-cp3{0}-cp3{0}m-linux_x86_64.whl'},
                  'windows': {'8': 'https://download.pytorch.org/whl/cu80/torch-1.0.0-cp3{0}-cp3{0}m-win_amd64.whl',
                              '9': 'https://download.pytorch.org/whl/cu90/torch-1.0.0-cp3{0}-cp3{0}m-win_amd64.whl',
                              '10': 'https://download.pytorch.org/whl/cu100/torch-1.0.0-cp3{0}-cp3{0}m-win_amd64.whl'}}

    _without_cuda = {'linux': 'https://download.pytorch.org/whl/cpu/torch-1.0.0-cp3{0}-cp3{0}m-linux_x86_64.whl',
                     'windows': 'https://download.pytorch.org/whl/cpu/torch-1.0.0-cp3{0}-cp3{0}m-win_amd64.whl'}

    def _installing_pytorch(os: str, cuda_version: str, installing_gpu: bool):
        print('---> Here')
        if (os.lower() != 'mac'):
            if (installing_gpu == 1):
                _dependencies.append(_with_cuda[os.lower()][cuda_version].format(_python_version))

            elif (installing_gpu == 0):
                _dependencies.append(_without_cuda[os.lower()].format(_python_version))

        elif (os.lower() == 'mac'):
            _dependencies.append('torch')


    _installing_gpu = int(input('Installing GPU or CPU version? (1 / 0): '))

    if (_installing_gpu == 1):
        _get_cuda_version = input('Choose Your Cuda Version (8 / 9 / 10): ')

    elif (_installing_gpu == 0):
        print('Installing Pytorch (CPU Version)')

    for dependent in _dependencies:
        os.system('pip3 install {0}'.format(dependent))
    os.system('pip3 install torchvision')

    sys.exit(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


class _ACNN(QMainWindow, acnn):
    clearDrawing = pyqtSignal()

    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.initUI()
        self.connecting_components()

    def initUI(self):
        self.canvas = PaintWidget(self)
        self.canvas.setObjectName('canvas')

        self.pred_button = QPushButton()
        self.pred_button.setText('Predict')

        self.clear_button = QPushButton()
        self.clear_button.setText('Clear')

        self.gridLayout.addWidget(self.canvas)
        self.gridLayout.addWidget(self.pred_button)
        self.gridLayout.addWidget(self.clear_button)

    def connecting_components(self):
        self.pred_button.clicked.connect(self.predict)
        self.clear_button.clicked.connect(self.clearCanvas)

    def clearCanvas(self):
        self.clearDrawing.emit()

    def preprocessing(self, x):
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize((28, 28))])
        return transform(x)

    def predict(self):
        model = CNN().to(device)
        model.load_state_dict(torch.load('model.ckpt'))
        model.eval()

        with torch.no_grad():
            _im = Image.fromarray(self.canvas.drawn_point.T)
            _im = self.preprocessing(_im)

            _im = np.reshape(_im, (1, 1, 28, 28))

            _im = torch.from_numpy(_im)
            _im = _im.type(torch.cuda.FloatTensor)

            _image = Variable(_im).to(device)
            result = model(_image)
            _, predicted = torch.max(result.data, 1)
            print(predicted)


class PaintWidget(QWidget):
    def __init__(self, _parent):
        super(PaintWidget, self).__init__()
        self.chosen_points = []
        self.drawn_point = np.zeros((self.width(), self.height()))
        self._parent = _parent
        self.isErase = False

        self.connecting_components()

    def mouseMoveEvent(self, event):
        if (event.x() < self.width() and event.y() < self.height()):
            self.chosen_points.append(event.pos())
            self.drawn_point[event.x(), event.y()] = 255
            # print(event.pos())
            self.update()

    def connecting_components(self):
        self._parent.clearDrawing.connect(self.clearCanvas)

    @pyqtSlot()
    def clearCanvas(self):
        self.drawn_point = np.zeros((self.width(), self.height()))
        self.chosen_points.clear()
        self.isErase = True

    def paintEvent(self, event):
        q = QPainter(self)
        p = QPen()
        p.setStyle(Qt.SolidLine)
        p.setWidth(5)
        q.setPen(p)

        if (self.isErase):
            q.eraseRect(self.rect())
            self.isErase = False

        for pos in self.chosen_points:
            q.drawPoint(pos)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = _ACNN()
    form.show()
    sys.exit(app.exec_())
