# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'myui.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(892, 720)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(30, 40, 711, 191))
        font = QtGui.QFont()
        font.setFamily("Arial Narrow")
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.pushButton11 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton11.setGeometry(QtCore.QRect(30, 40, 221, 46))
        self.pushButton11.setObjectName("pushButton11")
        self.pushButton12 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton12.setGeometry(QtCore.QRect(30, 120, 221, 46))
        self.pushButton12.setObjectName("pushButton12")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(280, 30, 421, 61))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setGeometry(QtCore.QRect(280, 110, 421, 61))
        self.label_3.setObjectName("label_3")
        self.textEdit = QtWidgets.QTextEdit(self.groupBox)
        self.textEdit.setGeometry(QtCore.QRect(400, 30, 71, 51))
        self.textEdit.setObjectName("textEdit")
        self.textEdit2 = QtWidgets.QTextEdit(self.groupBox)
        self.textEdit2.setGeometry(QtCore.QRect(400, 110, 71, 51))
        self.textEdit2.setObjectName("textEdit2")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(30, 230, 711, 251))
        font = QtGui.QFont()
        font.setFamily("Arial Narrow")
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.pushButton21 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton21.setGeometry(QtCore.QRect(30, 40, 261, 46))
        self.pushButton21.setObjectName("pushButton21")
        self.pushButton22 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton22.setGeometry(QtCore.QRect(30, 110, 261, 46))
        self.pushButton22.setObjectName("pushButton22")
        self.pushButton24 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton24.setGeometry(QtCore.QRect(30, 180, 261, 46))
        self.pushButton24.setObjectName("pushButton24")
        self.pushButton23 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton23.setGeometry(QtCore.QRect(340, 180, 261, 46))
        self.pushButton23.setObjectName("pushButton23")
        self.label = QtWidgets.QLabel(self.groupBox_2)
        self.label.setGeometry(QtCore.QRect(340, 70, 151, 24))
        self.label.setObjectName("label")
        self.comboBox = QtWidgets.QComboBox(self.groupBox_2)
        self.comboBox.setGeometry(QtCore.QRect(340, 120, 121, 30))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(30, 480, 331, 121))
        font = QtGui.QFont()
        font.setFamily("Arial Narrow")
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.groupBox_3.setFont(font)
        self.groupBox_3.setObjectName("groupBox_3")
        self.pushButton31 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton31.setGeometry(QtCore.QRect(30, 40, 261, 46))
        self.pushButton31.setObjectName("pushButton31")
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(360, 480, 371, 111))
        font = QtGui.QFont()
        font.setFamily("Arial Narrow")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_4.setFont(font)
        self.groupBox_4.setObjectName("groupBox_4")
        self.pushButton31_2 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton31_2.setGeometry(QtCore.QRect(20, 40, 331, 46))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.pushButton31_2.setFont(font)
        self.pushButton31_2.setObjectName("pushButton31_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 892, 36))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "1."))
        self.pushButton11.setText(_translate("MainWindow", "1.1 Draw contour"))
        self.pushButton12.setText(_translate("MainWindow", "1.2 Count Coins"))
        self.label_2.setText(_translate("MainWindow", "There are               coins in coin01.jpg"))
        self.label_3.setText(_translate("MainWindow", "There are               coins in coin02.jpg"))
        self.groupBox_2.setTitle(_translate("MainWindow", "2."))
        self.pushButton21.setText(_translate("MainWindow", "2.1 Corner detection "))
        self.pushButton22.setText(_translate("MainWindow", "2.2 Find the intrinsic "))
        self.pushButton24.setText(_translate("MainWindow", "2.4 Find the distortion "))
        self.pushButton23.setText(_translate("MainWindow", "2.3 Find the extrinsic "))
        self.label.setText(_translate("MainWindow", "Select Image"))
        self.comboBox.setItemText(0, _translate("MainWindow", "1"))
        self.comboBox.setItemText(1, _translate("MainWindow", "2"))
        self.comboBox.setItemText(2, _translate("MainWindow", "3"))
        self.comboBox.setItemText(3, _translate("MainWindow", "4"))
        self.comboBox.setItemText(4, _translate("MainWindow", "5"))
        self.comboBox.setItemText(5, _translate("MainWindow", "6"))
        self.comboBox.setItemText(6, _translate("MainWindow", "7"))
        self.comboBox.setItemText(7, _translate("MainWindow", "8"))
        self.comboBox.setItemText(8, _translate("MainWindow", "9"))
        self.comboBox.setItemText(9, _translate("MainWindow", "10"))
        self.comboBox.setItemText(10, _translate("MainWindow", "11"))
        self.comboBox.setItemText(11, _translate("MainWindow", "12"))
        self.comboBox.setItemText(12, _translate("MainWindow", "13"))
        self.comboBox.setItemText(13, _translate("MainWindow", "14"))
        self.comboBox.setItemText(14, _translate("MainWindow", "15"))
        self.comboBox.setItemText(15, _translate("MainWindow", "新增項目"))
        self.groupBox_3.setTitle(_translate("MainWindow", "3."))
        self.pushButton31.setText(_translate("MainWindow", "3.1 Augmented Reality "))
        self.groupBox_4.setTitle(_translate("MainWindow", "4."))
        self.pushButton31_2.setText(_translate("MainWindow", "4.1Compute disparity map"))




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())