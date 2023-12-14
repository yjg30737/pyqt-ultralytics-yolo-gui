import os
import sys

from script import YOLOWrapper, open_directory

# Get the absolute path of the current script file
script_path = os.path.abspath(__file__)

# Get the root directory by going up one level from the script directory
project_root = os.path.dirname(os.path.dirname(script_path))

sys.path.insert(0, project_root)
sys.path.insert(0, os.getcwd())  # Add the current directory as well

from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QGroupBox, QFormLayout, QComboBox, QCheckBox, QMessageBox
from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtGui import QFont

from PyQt5.QtCore import QThread, pyqtSignal

QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)  # HighDPI support

QApplication.setFont(QFont('Arial', 12))


class Thread(QThread):
    errorGenerated = pyqtSignal(str)
    generateFinished = pyqtSignal(str)

    def __init__(self, wrapper: YOLOWrapper, cur_task, path, plot_arg):
        super(Thread, self).__init__()
        self.__wrapper = wrapper
        self.__cur_task = cur_task
        self.__path = path
        self.__plot_arg = plot_arg

    def run(self):
        try:
            if os.path.exists(self.__path):
                dst_filename = self.__wrapper.get_result(self.__cur_task, self.__path, self.__plot_arg)
                self.generateFinished.emit(dst_filename)
            else:
                raise Exception(f'The file {self.__path} doesn\'t exists')
        except Exception as e:
            self.errorGenerated.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.__initVal()
        self.__initUi()

    def __initVal(self):
        self.__wrapper = YOLOWrapper()

    def __initUi(self):
        self.setWindowTitle('PyQt Ultralytics YOLO GUI')

        self.__btn = QPushButton('Run')
        self.__btn.clicked.connect(self.__run)

        self.__pathLineEdit = QLineEdit()
        self.__pathLineEdit.setPlaceholderText('Write the Full File Path (*.jpg, *.png, *.mp4)...')
        self.__pathLineEdit.textChanged.connect(self.__pathChanged)
        self.__pathLineEdit.returnPressed.connect(self.__btn.click)

        self.__taskCmbBox = QComboBox()
        self.__taskCmbBox.addItems(['Object Detection', 'Semantic Segmentation'])

        self.__boxesChkBox = QCheckBox()
        self.__labelsChkBox = QCheckBox()
        self.__confChkBox = QCheckBox()

        self.__boxesChkBox.setChecked(True)
        self.__labelsChkBox.setChecked(True)
        self.__confChkBox.setChecked(True)

        lay = QFormLayout()
        lay.addRow('Task', self.__taskCmbBox)
        lay.addRow('Show Boxes', self.__boxesChkBox)
        lay.addRow('Show Labels', self.__labelsChkBox)
        lay.addRow('Show Confidence', self.__confChkBox)

        settingsGrpBox = QGroupBox()
        settingsGrpBox.setTitle('Settings')
        settingsGrpBox.setLayout(lay)

        lay = QVBoxLayout()
        lay.addWidget(self.__pathLineEdit)
        lay.addWidget(self.__btn)
        lay.addWidget(settingsGrpBox)
        lay.setAlignment(Qt.AlignTop)

        mainWidget = QWidget()
        mainWidget.setLayout(lay)

        self.setCentralWidget(mainWidget)

        self.__btn.setEnabled(False)

    def __pathChanged(self, text):
        self.__btn.setEnabled(text.strip() != '')

    def __run(self):
        src_pathname = self.__pathLineEdit.text()
        cur_task = self.__taskCmbBox.currentIndex()

        is_boxes_checked = self.__boxesChkBox.isChecked()
        is_labels_checked = self.__labelsChkBox.isChecked()
        is_conf_checked = self.__confChkBox.isChecked()

        plot_arg = {
            'boxes': is_boxes_checked,
            'labels': is_labels_checked,
            'conf': is_conf_checked
        }

        self.__t = Thread(self.__wrapper, cur_task, src_pathname, plot_arg)
        self.__t.started.connect(self.__started)
        self.__t.errorGenerated.connect(self.__errorGenerated)
        self.__t.generateFinished.connect(self.__generatedFinished)
        self.__t.finished.connect(self.__finished)
        self.__t.start()
        self.__toggleWidget(False)

    def __toggleWidget(self, f):
        self.__boxesChkBox.setEnabled(f)
        self.__labelsChkBox.setEnabled(f)
        self.__confChkBox.setEnabled(f)
        self.__btn.setEnabled(f)

    def __started(self):
        print('started')

    def __errorGenerated(self, e):
        QMessageBox.critical(self, 'Error', e)

    def __generatedFinished(self, filename):
        open_directory(os.path.dirname(filename))

    def __finished(self):
        print('finished')
        self.__toggleWidget(True)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())