"""
IrisLens AI - AI-Powered Slit Lamp Expansion Module for Cataract Screening
Professional Minimalist Medical GUI Application with PyQt5
"""

import sys
import os
import base64
import json
import urllib.error
import urllib.request
from pathlib import Path

from PyQt5.QtCore import (
    Qt, QTimer, QRectF, QEvent, pyqtSignal, QPropertyAnimation, QLibraryInfo,
    QEasingCurve, QSize, QPoint
)
# OpenCV may set this to its own Qt plugin directory when imported. Point Qt
# at PyQt's matching plugins first so QApplication does not load the wrong xcb.
_qt_plugins = QLibraryInfo.location(QLibraryInfo.PluginsPath)
if _qt_plugins:
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _qt_plugins

import cv2
import numpy as np

# cv2 replaces this variable during import, so restore PyQt's plugin path.
if _qt_plugins:
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _qt_plugins

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, 
    QLineEdit, QComboBox, QVBoxLayout, QHBoxLayout, QGridLayout, 
    QStackedWidget, QFrame, QProgressBar, QMessageBox, 
    QGraphicsDropShadowEffect, QDialog, QSizePolicy, QSpacerItem, QScrollArea
)
from PyQt5.QtGui import (
    QFont, QColor, QPainter, QPen, QBrush, QPixmap, 
    QImage, QIcon, QPainterPath
)
from ncnn_detector import NCNNDetector

# --- OFFICIAL COLOR PALETTE ---
COLOR_BG = "#F4F7F6"          # Clinical Light Grey-White
COLOR_NAVY = "#0B2545"        # Deep Navy
COLOR_NAVY_HOVER = "#12375F"  # Navy hover
COLOR_CYAN = "#00D2C8"        # Vibrant Bright Cyan
COLOR_CYAN_HOVER = "#00B8AF"  # Cyan hover
COLOR_AMBER = "#E68200"       # Amber-Orange Badge (Partial Cataract)
COLOR_GREY_BTN = "#6D7C8A"    # Secondary Button Grey
COLOR_GREY_HOVER = "#586774"  # Secondary Button Grey Hover
COLOR_CARD_BOX = "#E2E8EE"    # Form card box grey
COLOR_CARD_BORDER = "#D0D9E0" # Form card border
COLOR_TEXT_NAVY = "#0B2545"   # Primary text navy
COLOR_TEXT_MUTED = "#556877"  # Form label grey
COLOR_WHITE = "#FFFFFF"       # Crisp White

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")


class AnimatedLoadingRing(QWidget):
    """
    Eye and slit-lamp logo widget with a smoothly rotating cyan loading ring (60 FPS).
    As specified for Page 1: Centered logo featuring a sleek minimalist eye and 
    slit-lamp lens symbol in deep navy #0B2545 surrounded by a cyan animated circular loading ring #00D2C8.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(290, 290)
        self.angle = 0
        
        # Load inner eye symbol
        symbol_path = os.path.join(ASSETS_DIR, "eye_symbol.png")
        if os.path.exists(symbol_path):
            self.eye_pixmap = QPixmap(symbol_path).scaled(
                180, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        else:
            self.eye_pixmap = None
            
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_tick)
        self.timer.start(16)  # ~60 FPS
        
    def on_tick(self):
        self.angle = (self.angle + 3.0) % 360
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        cx = self.width() / 2.0
        cy = self.height() / 2.0
        radius = 105.0
        rect = QRectF(cx - radius, cy - radius, radius * 2, radius * 2)
        
        # 1. Subtle static background circle track
        track_pen = QPen(QColor(0, 210, 200, 40), 5.5)
        painter.setPen(track_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(rect)
        
        # 2. Primary rotating cyan arc
        arc_pen = QPen(QColor(0, 210, 200, 255), 5.5, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(arc_pen)
        start_angle = int(self.angle * 16)
        span_angle = int(120 * 16)
        painter.drawArc(rect, start_angle, span_angle)
        
        # Secondary shorter decorative arc opposite
        span_angle2 = int(45 * 16)
        painter.drawArc(rect, int((self.angle + 175) * 16), span_angle2)
        
        # 3. Draw Center Eye Symbol
        if self.eye_pixmap and not self.eye_pixmap.isNull():
            px = int(cx - self.eye_pixmap.width() / 2.0)
            py = int(cy - self.eye_pixmap.height() / 2.0)
            painter.drawPixmap(px, py, self.eye_pixmap)


class SplashScreen(QWidget):
    """
    Page 1: Splash Loading Screen.
    A clean minimalist medical UI design for a splash loading screen on a 1920x1080 display.
    Light clinical background #F4F7F6, Centered logo with cyan animated ring #00D2C8,
    Title 'IrisLens AI', Subtitle 'Smart Optics for Precision Cataract Screening',
    Bottom 'AI-Powered Slit Lamp Expansion Module for Cataract Screening'.
    """
    navigate_to_registration = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {COLOR_BG};")
        self.init_ui()
        
        # Automatically transition to the registration page after 3.5 seconds
        self.auto_timer = QTimer(self)
        self.auto_timer.setSingleShot(True)
        self.auto_timer.timeout.connect(self.go_next)
        self.auto_timer.start(3500)
        
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 60, 40, 55)
        main_layout.setSpacing(0)
        
        main_layout.addStretch(3)
        
        # Centered container
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setSpacing(10)
        center_layout.setAlignment(Qt.AlignCenter)
        
        # Animated Logo
        self.loading_logo = AnimatedLoadingRing(self)
        center_layout.addWidget(self.loading_logo, 0, Qt.AlignCenter)
        
        center_layout.addSpacing(18)
        
        # App Title: "IrisLens AI" ("IrisLens" in deep navy #0B2545, "AI" in bright cyan #00D2C8)
        title_label = QLabel()
        title_label.setText(
            f'<span style="color: {COLOR_NAVY}; font-weight: 800; font-family: Segoe UI, sans-serif;">IrisLens</span> '
            f'<span style="color: {COLOR_CYAN}; font-weight: 800; font-family: Segoe UI, sans-serif;">AI</span>'
        )
        title_label.setFont(QFont("Segoe UI", 38, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        center_layout.addWidget(title_label)
        
        # Subtitle: "Smart Optics for Precision Cataract Screening"
        sub_label = QLabel("Smart Optics for Precision Cataract Screening")
        sub_label.setFont(QFont("Segoe UI", 16, QFont.Normal))
        sub_label.setStyleSheet(f"color: {COLOR_NAVY}; letter-spacing: 0.3px;")
        sub_label.setAlignment(Qt.AlignCenter)
        center_layout.addWidget(sub_label)
        
        main_layout.addWidget(center_widget, 0, Qt.AlignCenter)
        main_layout.addStretch(4)
        
        # Bottom Center: "AI-Powered Slit Lamp Expansion Module for Cataract Screening"
        bottom_label = QLabel("AI-Powered Slit Lamp Expansion Module for Cataract Screening")
        bottom_label.setFont(QFont("Segoe UI", 13, QFont.Normal))
        bottom_label.setStyleSheet(f"color: {COLOR_NAVY};")
        bottom_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(bottom_label, 0, Qt.AlignCenter)
        
    def go_next(self):
        self.auto_timer.stop()
        self.navigate_to_registration.emit()
        
    def mousePressEvent(self, event):
        # Click anywhere to skip the splash screen
        self.go_next()


class RegistrationScreen(QWidget):
    """
    Page 2: New Patient Registration.
    Top header bar in deep navy #0B2545 with "IrisLens AI" on left and "New Patient Registration" in center.
    Card container with 4 form input fields: "PATIENT ID", "PATIENT NAME", "AGE", "GENDER".
    Two buttons at bottom right: "CANCEL / RESET" and "CONTINUE TO EXAMINATION".
    """
    navigate_to_examination = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {COLOR_BG};")
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 1. Top Header Bar (Full width, Deep Navy #0B2545)
        header = QFrame()
        header.setFixedHeight(72)
        header.setStyleSheet(f"background-color: {COLOR_NAVY};")
        
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 0, 24, 0)
        
        # Left: IrisLens AI (Clean White)
        app_brand = QLabel("IrisLens AI")
        app_brand.setFont(QFont("Segoe UI", 20, QFont.Bold))
        app_brand.setStyleSheet(f"color: {COLOR_WHITE};")
        header_layout.addWidget(app_brand)
        
        header_layout.addStretch(1)
        
        # Center: New Patient Registration
        page_title = QLabel("New Patient Registration")
        page_title.setFont(QFont("Segoe UI", 17, QFont.DemiBold))
        page_title.setStyleSheet(f"color: {COLOR_WHITE};")
        page_title.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(page_title)
        
        header_layout.addStretch(1)
        
        # Dummy spacer to balance center alignment
        spacer = QLabel()
        spacer.setFixedWidth(app_brand.sizeHint().width())
        header_layout.addWidget(spacer)
        
        main_layout.addWidget(header)
        
        # 2. Main Content Area
        content_wrapper = QWidget()
        content_layout = QVBoxLayout(content_wrapper)
        content_layout.setContentsMargins(32, 24, 32, 24)
        content_layout.setAlignment(Qt.AlignCenter)
        
        # Form Container Box (Centered)
        form_box = QWidget()
        form_box.setMaximumWidth(900)
        form_inner_layout = QVBoxLayout(form_box)
        form_inner_layout.setSpacing(18)
        form_inner_layout.setContentsMargins(0, 0, 0, 0)
        
        # 2x2 Grid of Form Fields
        grid_layout = QGridLayout()
        grid_layout.setHorizontalSpacing(20)
        grid_layout.setVerticalSpacing(16)
        
        # Field 1: PATIENT ID
        self.card_id, self.input_id = self.create_form_card("PATIENT ID")
        self.input_id.setPlaceholderText("PATIENT ID...")
        grid_layout.addWidget(self.card_id, 0, 0)
        
        # Field 2: PATIENT NAME
        self.card_name, self.input_name = self.create_form_card("PATIENT NAME")
        self.input_name.setPlaceholderText("Enter the patient’s full name...")
        grid_layout.addWidget(self.card_name, 0, 1)
        
        # Field 3: AGE
        self.card_age, self.input_age = self.create_form_card("AGE")
        self.input_age.setPlaceholderText("PATIENT AGE...")
        grid_layout.addWidget(self.card_age, 1, 0)
        
        # Field 4: GENDER
        self.card_gender, self.combo_gender = self.create_gender_card("GENDER")
        grid_layout.addWidget(self.card_gender, 1, 1)
        
        form_inner_layout.addLayout(grid_layout)
        
        # Buttons Row (Bottom Right of the form container)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(18)
        btn_row.addStretch()
        
        # CANCEL / RESET Button (Grey)
        self.btn_reset = QPushButton("CANCEL / RESET")
        self.btn_reset.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.btn_reset.setCursor(Qt.PointingHandCursor)
        self.btn_reset.setFixedHeight(44)
        self.btn_reset.setMinimumWidth(160)
        self.btn_reset.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_GREY_BTN};
                color: {COLOR_WHITE};
                border: none;
                border-radius: 9px;
                padding: 12px 24px;
            }}
            QPushButton:hover {{
                background-color: {COLOR_GREY_HOVER};
            }}
            QPushButton:pressed {{
                background-color: #4C5963;
            }}
        """)
        self.btn_reset.clicked.connect(self.on_reset)
        btn_row.addWidget(self.btn_reset)
        
        # CONTINUE TO EXAMINATION Button (Vibrant Cyan #00D2C8)
        self.btn_submit = QPushButton("CONTINUE TO EXAMINATION")
        self.btn_submit.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.btn_submit.setCursor(Qt.PointingHandCursor)
        self.btn_submit.setFixedHeight(44)
        self.btn_submit.setMinimumWidth(250)
        self.btn_submit.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_CYAN};
                color: {COLOR_WHITE};
                border: none;
                border-radius: 9px;
                padding: 12px 28px;
            }}
            QPushButton:hover {{
                background-color: {COLOR_CYAN_HOVER};
            }}
            QPushButton:pressed {{
                background-color: #009E96;
            }}
        """)
        self.btn_submit.clicked.connect(self.on_submit)
        btn_row.addWidget(self.btn_submit)
        
        form_inner_layout.addLayout(btn_row)
        
        content_layout.addWidget(form_box)

        # Keep the active field reachable when the on-screen keyboard is open.
        self.form_scroll = QScrollArea()
        self.form_scroll.setWidgetResizable(True)
        self.form_scroll.setFrameShape(QFrame.NoFrame)
        self.form_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.form_scroll.setWidget(content_wrapper)
        main_layout.addWidget(self.form_scroll, 1)

        # A keyboard inside the app avoids relying on focus/input forwarding
        # from the desktop OSK (which can vary between Wayland and XWayland).
        self.virtual_keyboard = RegistrationKeyboard(self)
        self.virtual_keyboard.hide()
        main_layout.addWidget(self.virtual_keyboard)

        for field in (self.input_id, self.input_name, self.input_age):
            field.setFocusPolicy(Qt.StrongFocus)
            field.installEventFilter(self)
            field.textEdited.connect(self.virtual_keyboard.update_caps_state)

    def create_form_card(self, label_text):
        """Create a grey card container with a label and white input field matching the mockup."""
        card = QFrame()
        card.setFixedHeight(120)
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {COLOR_CARD_BOX};
                border: 1.5px solid {COLOR_CARD_BORDER};
                border-radius: 14px;
            }}
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 14, 22, 16)
        layout.setSpacing(8)
        
        lbl = QLabel(label_text)
        lbl.setFont(QFont("Segoe UI", 11, QFont.DemiBold))
        lbl.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; border: none; background: transparent; letter-spacing: 0.4px;")
        layout.addWidget(lbl)
        
        inp = QLineEdit()
        inp.setFont(QFont("Segoe UI", 12))
        inp.setFixedHeight(46)
        inp.setStyleSheet(f"""
            QLineEdit {{
                background-color: {COLOR_WHITE};
                color: {COLOR_NAVY};
                border: 1px solid #D0D9E0;
                border-radius: 8px;
                padding-left: 14px;
                padding-right: 14px;
            }}
            QLineEdit:focus {{
                border: 1.5px solid {COLOR_CYAN};
            }}
        """)
        layout.addWidget(inp)
        return card, inp

    def eventFilter(self, watched, event):
        if watched in (self.input_id, self.input_name, self.input_age) and event.type() == QEvent.FocusIn:
            self.virtual_keyboard.set_target(watched)
            self.virtual_keyboard.show()
            QTimer.singleShot(0, lambda field=watched: self._scroll_to_field(field))
        return super().eventFilter(watched, event)

    def _scroll_to_field(self, field):
        self.form_scroll.ensureWidgetVisible(field, 20, 20)

    def hide_virtual_keyboard(self):
        self.virtual_keyboard.hide()

    def focusNextTextField(self):
        fields = (self.input_id, self.input_name, self.input_age)
        try:
            next_field = fields[(fields.index(self.virtual_keyboard.target) + 1) % len(fields)]
        except (ValueError, AttributeError):
            next_field = self.input_id
        next_field.setFocus(Qt.TabFocusReason)

    def create_gender_card(self, label_text):
        """Create a gender card with a modern combo box."""
        card = QFrame()
        card.setFixedHeight(120)
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {COLOR_CARD_BOX};
                border: 1.5px solid {COLOR_CARD_BORDER};
                border-radius: 14px;
            }}
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 14, 22, 16)
        layout.setSpacing(8)
        
        lbl = QLabel(label_text)
        lbl.setFont(QFont("Segoe UI", 11, QFont.DemiBold))
        lbl.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; border: none; background: transparent; letter-spacing: 0.4px;")
        layout.addWidget(lbl)
        
        combo = QComboBox()
        combo.addItems(["Select gender", "Male", "Female"])
        combo.setFont(QFont("Segoe UI", 12))
        combo.setFixedHeight(46)
        combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {COLOR_WHITE};
                color: {COLOR_NAVY};
                border: 1px solid #D0D9E0;
                border-radius: 8px;
                padding-left: 14px;
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 35px;
                border-left-width: 0px;
                border-top-right-radius: 8px;
                border-bottom-right-radius: 8px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {COLOR_WHITE};
                selection-background-color: {COLOR_CYAN};
                selection-color: {COLOR_WHITE};
                color: {COLOR_NAVY};
                border: 1px solid #D0D9E0;
                border-radius: 6px;
                padding: 4px;
            }}
        """)
        layout.addWidget(combo)
        return card, combo

    def on_reset(self):
        self.input_id.clear()
        self.input_name.clear()
        self.input_age.clear()
        self.combo_gender.setCurrentIndex(0)

    def on_submit(self):
        name = self.input_name.text().strip()
        patient_id = self.input_id.text().strip()
        age = self.input_age.text().strip()
        gender = "" if self.combo_gender.currentIndex() == 0 else self.combo_gender.currentText()

        missing = []
        if not patient_id:
            missing.append(("patient ID", self.input_id))
        if not name:
            missing.append(("patient name", self.input_name))
        if not age:
            missing.append(("age", self.input_age))
        if not gender:
            missing.append(("gender", self.combo_gender))
        if missing:
            field_name, field = missing[0]
            QMessageBox.warning(self, "Incomplete information", f"Please enter the {field_name} before continuing.")
            field.setFocus()
            return
        
        data = {
            "id": patient_id,
            "name": name,
            "age": age,
            "gender": gender
        }
        self.navigate_to_examination.emit(data)


class RegistrationKeyboard(QWidget):
    """Touch keyboard for the registration fields; buttons do not take focus away from the fields."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.target = None
        self.caps = True
        self.setObjectName("registrationKeyboard")
        self.setStyleSheet(f"""
            QWidget#registrationKeyboard {{ background: {COLOR_NAVY}; }}
            QPushButton {{ background: #24415F; color: white; border: 1px solid #54708B;
                border-radius: 6px; min-height: 38px; font-size: 16px; font-weight: 600; }}
            QPushButton:pressed {{ background: {COLOR_CYAN}; color: {COLOR_NAVY}; }}
        """)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(4)
        rows = ("1234567890", "qwertyuiop", "asdfghjkl", "zxcvbnm")
        self.letter_buttons = []
        for row_index, chars in enumerate(rows):
            row = QHBoxLayout()
            row.setSpacing(4)
            if row_index == 3:
                caps_button = self._button("⇧", lambda: self._toggle_caps())
                row.addWidget(caps_button, 1)
            for char in chars:
                button = self._button(char, lambda value=char: self._insert(value))
                row.addWidget(button, 1)
                if char.isalpha():
                    self.letter_buttons.append((button, char))
            if row_index == 3:
                row.addWidget(self._button("⌫", self._backspace), 1)
            outer.addLayout(row)
        bottom = QHBoxLayout()
        bottom.setSpacing(4)
        bottom.addWidget(self._button("-", lambda: self._insert("-")), 1)
        bottom.addWidget(self._button("_", lambda: self._insert("_")), 1)
        bottom.addWidget(self._button("Space", lambda: self._insert(" ")), 5)
        bottom.addWidget(self._button("Done", self._finish), 2)
        outer.addLayout(bottom)
        self.setFixedHeight(222)

    def _button(self, label, callback):
        button = QPushButton(label)
        button.setFocusPolicy(Qt.NoFocus)
        button.clicked.connect(lambda checked=False: callback())
        return button

    def set_target(self, field):
        self.target = field
        self.update_caps_state()

    def update_caps_state(self, *_):
        if self.target is not None and self.target.text():
            self.caps = False
        for button, char in self.letter_buttons:
            button.setText(char.upper() if self.caps else char.lower())

    def _toggle_caps(self):
        self.caps = not self.caps
        self.update_caps_state()

    def _insert(self, value):
        if self.target is None:
            return
        self.target.insert(value.upper() if value.isalpha() and self.caps else value)

    def _backspace(self):
        if self.target is None:
            return
        self.target.backspace()

    def _finish(self):
        if self.target is not None and self.target is not self.parent().input_age:
            self.parent().focusNextTextField()
        else:
            self.hide()

class ExaminationScreen(QWidget):
    """
    Page 3: Medical Examination Dashboard UI.
    Top header bar in deep navy #0B2545 displaying "IrisLens AI".
    Left Panel: Video preview of eye pupil under slit-lamp lighting,
      cyan badge "AI: MATA KANAN (OD) DETECTED", control buttons "START", "STOP", "CAPTURE".
    Right Panel: Clean white card displaying YOLO26 prediction and confidence,
      button "SEND TO DOCTOR".
    """
    back_to_registration = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {COLOR_BG};")
        
        self.patient_info = {"id": "", "name": "", "age": "", "gender": ""}
        
        # Camera and stream state
        self.camera_indices = self.find_camera_indices()
        self.camera_index = self.camera_indices[0] if self.camera_indices else None
        self.cap = cv2.VideoCapture(self.camera_index) if self.camera_index is not None else None
        self.is_streaming = True
        self.current_frame_bgr = None
        self.last_prediction = None
        self.last_capture_bgr = None
        self.last_detections = []

        model_dir = Path(os.environ.get(
            "IRISLENS_NCNN_MODEL",
            str(Path(__file__).resolve().parents[2] / "train" / "runs" / "cataract-yolo26n" / "weights" / "best_ncnn_model"),
        )).expanduser().resolve()
        self.model_error = None
        try:
            self.model = NCNNDetector(model_dir)
        except Exception as exc:
            self.model = None
            self.model_error = str(exc)
        
        self.init_ui()
        
        # Timer untuk streaming video (30 FPS)
        self.stream_timer = QTimer(self)
        self.stream_timer.timeout.connect(self.update_video_frame)
        self.stream_timer.start(33)

    @staticmethod
    def find_camera_indices(max_index=5):
        available = []
        for index in range(max_index):
            camera = cv2.VideoCapture(index)
            if camera.isOpened():
                available.append(index)
            camera.release()
        return available

    def set_patient_data(self, data):
        self.patient_info = data
        self.patient_chip.setText(
            f"ID: <b>{data['id'] or '—'}</b> &nbsp; Nama: <b>{data['name'] or '—'}</b> &nbsp; "
            f"Age: <b>{data['age'] or '—'}</b> &nbsp; Gender: <b>{data['gender'] or '—'}</b>"
        )

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(18, 8, 18, 10)
        main_layout.setSpacing(8)
        
        # 1. Top Header Bar (Centered Deep Navy #0B2545 Card matching reference)
        header_card = QFrame()
        header_card.setFixedHeight(52)
        header_card.setStyleSheet(f"""
            QFrame {{
                background-color: {COLOR_NAVY};
                border-radius: 12px;
            }}
        """)
        header_layout = QHBoxLayout(header_card)
        header_layout.setContentsMargins(14, 0, 14, 0)
        
        # Back button
        self.btn_back = QPushButton("← New Registration")
        self.btn_back.setFont(QFont("Segoe UI", 10, QFont.DemiBold))
        self.btn_back.setCursor(Qt.PointingHandCursor)
        self.btn_back.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(255, 255, 255, 0.12);
                color: {COLOR_WHITE};
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 6px;
                padding: 6px 14px;
            }}
            QPushButton:hover {{
                background-color: rgba(255, 255, 255, 0.25);
            }}
        """)
        self.btn_back.clicked.connect(self.on_back_clicked)
        header_layout.addWidget(self.btn_back)
        
        header_layout.addStretch(1)
        
        # Centered "IrisLens AI"
        app_brand = QLabel("IrisLens AI")
        app_brand.setFont(QFont("Segoe UI", 18, QFont.Bold))
        app_brand.setStyleSheet(f"color: {COLOR_WHITE};")
        app_brand.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(app_brand)
        
        header_layout.addStretch(1)
        
        # Right: Patient Info Chip
        self.patient_chip = QLabel("Patient information will appear here")
        self.patient_chip.setFont(QFont("Segoe UI", 8))
        self.patient_chip.setMaximumWidth(430)
        self.patient_chip.setStyleSheet(f"""
            color: {COLOR_WHITE};
            background-color: rgba(0, 210, 200, 0.18);
            border: 1px solid rgba(0, 210, 200, 0.4);
            border-radius: 6px;
            padding: 6px 14px;
        """)
        header_layout.addWidget(self.patient_chip)
        
        main_layout.addWidget(header_card)
        
        # 2. Main Content: Split into Two Panels (Left & Right)
        panels_layout = QHBoxLayout()
        panels_layout.setSpacing(12)
        
        # ========================================================
        # LEFT PANEL: Video Preview & Controls
        # ========================================================
        left_panel = QFrame()
        left_panel.setStyleSheet(f"""
            QFrame {{
                background-color: {COLOR_WHITE};
                border-radius: 14px;
                border: 1px solid #E2E8F0;
            }}
        """)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 8, 10, 8)
        left_layout.setSpacing(7)
        
        # Video Container Box (Black background)
        self.video_box = QWidget()
        self.video_box.setStyleSheet("background-color: #000000; border-radius: 10px;")
        video_box_layout = QVBoxLayout(self.video_box)
        video_box_layout.setContentsMargins(0, 0, 0, 0)
        
        # Video Label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("border-radius: 10px; overflow: hidden;")
        video_box_layout.addWidget(self.video_label)
        
        # Translucent Cyan Badge in Top-Left Corner
        self.ai_badge = QLabel(self.video_box)
        self.ai_badge.setText("MODEL YOLO26")
        self.ai_badge.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.ai_badge.setStyleSheet(f"""
            background-color: {COLOR_CYAN};
            color: {COLOR_WHITE};
            border-radius: 6px;
            padding: 7px 16px;
            letter-spacing: 0.5px;
        """)
        self.ai_badge.move(20, 20)
        self.ai_badge.show()
        
        left_layout.addWidget(self.video_box, 1)
        
        # Control Buttons Row below Video
        btn_controls_layout = QHBoxLayout()
        btn_controls_layout.setSpacing(16)
        
        # 1. START Button (Cyan #00D2C8)
        self.btn_mulai = QPushButton("START")
        self.btn_mulai.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.btn_mulai.setCursor(Qt.PointingHandCursor)
        self.btn_mulai.setFixedHeight(40)
        self.btn_mulai.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_CYAN};
                color: {COLOR_WHITE};
                border: none;
                border-radius: 8px;
                letter-spacing: 0.5px;
            }}
            QPushButton:hover {{
                background-color: {COLOR_CYAN_HOVER};
            }}
            QPushButton:pressed {{
                background-color: #009E96;
            }}
        """)
        self.btn_mulai.clicked.connect(self.on_start_stream)
        btn_controls_layout.addWidget(self.btn_mulai, 1)
        
        # 2. STOP Button (White with Navy border)
        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.btn_stop.setCursor(Qt.PointingHandCursor)
        self.btn_stop.setFixedHeight(40)
        self.btn_stop.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_WHITE};
                color: {COLOR_NAVY};
                border: 1.5px solid {COLOR_NAVY};
                border-radius: 8px;
                letter-spacing: 0.5px;
            }}
            QPushButton:hover {{
                background-color: #F0F4F8;
            }}
            QPushButton:pressed {{
                background-color: #E2E8F0;
            }}
        """)
        self.btn_stop.clicked.connect(self.on_stop_stream)
        btn_controls_layout.addWidget(self.btn_stop, 1)
        
        # 3. CAPTURE Button (Deep Navy #0B2545)
        self.btn_capture = QPushButton("CAPTURE")
        self.btn_capture.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.btn_capture.setCursor(Qt.PointingHandCursor)
        self.btn_capture.setFixedHeight(40)
        self.btn_capture.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_NAVY};
                color: {COLOR_WHITE};
                border: none;
                border-radius: 8px;
                letter-spacing: 0.5px;
            }}
            QPushButton:hover {{
                background-color: {COLOR_NAVY_HOVER};
            }}
            QPushButton:pressed {{
                background-color: #05162B;
            }}
        """)
        self.btn_capture.clicked.connect(self.on_capture)
        btn_controls_layout.addWidget(self.btn_capture, 1)
        
        left_layout.addLayout(btn_controls_layout)

        camera_source_row = QHBoxLayout()
        camera_source_label = QLabel("Camera source:")
        camera_source_label.setStyleSheet("color: #64748B; border: none;")
        camera_source_row.addWidget(camera_source_label)
        self.camera_combo = QComboBox()
        self.camera_combo.setStyleSheet(f"""
            QComboBox {{
                border: 1px solid #CBD5E1;
                border-radius: 5px;
                padding: 4px 8px;
                color: {COLOR_NAVY};
                background: white;
            }}
        """)
        if self.camera_indices:
            for camera_index in self.camera_indices:
                self.camera_combo.addItem(f"Camera {camera_index}", camera_index)
        else:
            self.camera_combo.addItem("No camera detected", -1)
            self.camera_combo.setEnabled(False)
        self.camera_combo.currentIndexChanged.connect(self.on_camera_changed)
        camera_source_row.addWidget(self.camera_combo)
        camera_source_row.addStretch()
        left_layout.addLayout(camera_source_row)
        
        panels_layout.addWidget(left_panel, 65)
        
        # ========================================================
        # RIGHT PANEL: AI Analysis Results Card
        # ========================================================
        right_panel = QFrame()
        right_panel.setMinimumWidth(270)
        right_panel.setMaximumWidth(320)
        right_panel.setStyleSheet(f"""
            QFrame {{
                background-color: {COLOR_WHITE};
                border-radius: 14px;
                border: 1px solid #E2E8F0;
            }}
        """)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(16, 10, 16, 10)
        right_layout.setSpacing(7)
        
        # Title: AI DETECTION RESULTS
        lbl_ai_title = QLabel("AI DETECTION RESULTS")
        lbl_ai_title.setFont(QFont("Segoe UI", 12, QFont.DemiBold))
        lbl_ai_title.setStyleSheet("color: #4A5568; letter-spacing: 1px; border: none;")
        lbl_ai_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(lbl_ai_title)
        
        # Subtitle: MATA KANAN (OD)
        self.lbl_eye_side = QLabel("YOLO26 RESULTS")
        self.lbl_eye_side.setFont(QFont("Segoe UI", 16, QFont.Bold))
        self.lbl_eye_side.setStyleSheet(f"color: {COLOR_NAVY}; border: none;")
        self.lbl_eye_side.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.lbl_eye_side)
        
        right_layout.addSpacing(2)
        
        # Prediction badge, updated from the YOLO26 result after capture.
        self.badge_diagnosis = QLabel("NOT ANALYZED")
        self.badge_diagnosis.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.badge_diagnosis.setAlignment(Qt.AlignCenter)
        self.badge_diagnosis.setFixedHeight(44)
        self.badge_diagnosis.setStyleSheet(f"""
            background-color: {COLOR_AMBER};
            color: {COLOR_WHITE};
            border-radius: 8px;
            letter-spacing: 0.5px;
            border: none;
        """)
        right_layout.addWidget(self.badge_diagnosis)
        
        right_layout.addSpacing(4)
        
        # Confidence Score Row
        score_header = QHBoxLayout()
        lbl_score = QLabel("Confidence Score")
        lbl_score.setFont(QFont("Segoe UI", 12))
        lbl_score.setStyleSheet("color: #334155; border: none;")
        score_header.addWidget(lbl_score)
        
        score_header.addStretch()
        
        self.lbl_score_val = QLabel("—")
        self.lbl_score_val.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.lbl_score_val.setStyleSheet(f"color: {COLOR_NAVY}; border: none;")
        score_header.addWidget(self.lbl_score_val)
        right_layout.addLayout(score_header)
        
        # Progress Bar (Cyan fill #00D2C8, Grey track)
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(10)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: #E2E8F0;
                border-radius: 5px;
                border: none;
            }}
            QProgressBar::chunk {{
                background-color: {COLOR_CYAN};
                border-radius: 5px;
            }}
        """)
        right_layout.addWidget(self.progress_bar)
        
        # Clinical Metadata Summary
        right_layout.addSpacing(4)
        clinical_box = QFrame()
        clinical_box.setStyleSheet("""
            background-color: #F8FAFC;
            border: 1px solid #EDF2F7;
            border-radius: 8px;
            padding: 10px;
        """)
        c_layout = QVBoxLayout(clinical_box)
        c_layout.setSpacing(6)
        c_layout.setContentsMargins(12, 10, 12, 10)
        
        c_title = QLabel("Screening Analysis Details:")
        c_title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        c_title.setStyleSheet("color: #475569; border: none; background: transparent;")
        c_layout.addWidget(c_title)
        
        self.result_detail1 = QLabel("• Model class: waiting for capture")
        self.result_detail1.setFont(QFont("Segoe UI", 9))
        self.result_detail1.setStyleSheet("color: #64748B; border: none; background: transparent;")
        c_layout.addWidget(self.result_detail1)
        
        self.result_detail2 = QLabel("• Detection: not run")
        self.result_detail2.setFont(QFont("Segoe UI", 9))
        self.result_detail2.setStyleSheet("color: #64748B; border: none; background: transparent;")
        c_layout.addWidget(self.result_detail2)
        
        right_layout.addWidget(clinical_box)
        
        right_layout.addStretch(1)
        
        # Large Vibrant Cyan Action Button: SEND TO DOCTOR
        self.btn_send = QPushButton("SEND TO DOCTOR")
        self.btn_send.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.btn_send.setCursor(Qt.PointingHandCursor)
        self.btn_send.setFixedHeight(44)
        self.btn_send.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_CYAN};
                color: {COLOR_WHITE};
                border: none;
                border-radius: 8px;
                letter-spacing: 0.5px;
            }}
            QPushButton:hover {{
                background-color: {COLOR_CYAN_HOVER};
            }}
            QPushButton:pressed {{
                background-color: #009E96;
            }}
        """)
        self.btn_send.clicked.connect(self.on_send_to_doctor)
        right_layout.addWidget(self.btn_send)
        
        panels_layout.addWidget(right_panel, 35)
        
        main_layout.addLayout(panels_layout, 1)

    def on_camera_changed(self, combo_index):
        selected_camera = self.camera_combo.itemData(combo_index)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.camera_index = selected_camera if selected_camera is not None and selected_camera >= 0 else None
        self.cap = cv2.VideoCapture(self.camera_index) if self.camera_index is not None else None
        self.current_frame_bgr = None
        self.is_streaming = True

    def on_start_stream(self):
        if self.camera_index is not None and (self.cap is None or not self.cap.isOpened()):
            self.cap = cv2.VideoCapture(self.camera_index)
        self.is_streaming = True
        self.ai_badge.show()

    def on_stop_stream(self):
        self.is_streaming = False

    def on_capture(self):
        if self.current_frame_bgr is None:
            QMessageBox.warning(self, "Frame Not Available", "Wait for the camera image to appear, then try capturing again.")
            return
        if self.model is None:
            QMessageBox.critical(
                self,
                "YOLO26 NCNN Model Unavailable",
                f"{self.model_error or 'NCNN model is not configured.'}\n\n"
                "Set IRISLENS_NCNN_MODEL to the NCNN model folder containing model.ncnn.param and model.ncnn.bin.",
            )
            return

        try:
            detections = self.model.predict(self.current_frame_bgr)
        except Exception as exc:
            QMessageBox.critical(self, "Inference Failed", f"The model could not analyze the image:\n{exc}")
            return

        self.is_streaming = False
        self.last_capture_bgr = self.current_frame_bgr.copy()
        self.last_detections = detections
        self.last_prediction = "NOT DETECTED (0%)"
        # Keep the capture preview clean; model annotations and labels stay in the result panel.
        self.ai_badge.hide()
        self.video_label.setPixmap(QPixmap.fromImage(self._frame_to_qimage(self.current_frame_bgr)).scaled(
            self.video_label.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
        ))
        self.lbl_eye_side.setText("YOLO26 RESULTS")

        if not detections:
            self.ai_badge.setText("NO OBJECTS DETECTED")
            self.badge_diagnosis.setText("NOT DETECTED")
            self.lbl_score_val.setText("0%")
            self.progress_bar.setValue(0)
            self.result_detail1.setText("• Model class: not found")
            self.result_detail2.setText("• Detection: no objects")
            return

        best_detection = max(detections, key=lambda detection: detection["confidence"])
        class_name = best_detection["class_name"]
        confidence = best_detection["confidence"]
        confidence_percent = round(confidence * 100)
        translated = {
            "immature": "IMMATURE CATARACT",
            "mature": "MATURE CATARACT",
            "normal": "NORMAL",
        }.get(class_name.lower(), class_name.upper())

        self.last_prediction = f"{translated} ({confidence_percent}%)"
        self.ai_badge.setText(f"DETECTION: {class_name.upper()}")
        self.badge_diagnosis.setText(translated)
        self.lbl_score_val.setText(f"{confidence_percent}%")
        self.progress_bar.setValue(confidence_percent)
        self.result_detail1.setText(f"• Model class: {class_name}")
        self.result_detail2.setText(f"• Object count: {len(detections)}")

    def on_send_to_doctor(self):
        if self.last_capture_bgr is None or self.last_prediction is None:
            QMessageBox.information(self, "No Results Yet", "Press CAPTURE to run the model first.")
            return
        ok, encoded = cv2.imencode(".jpg", self.last_capture_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            QMessageBox.critical(self, "Send Failed", "The captured image could not be prepared.")
            return
        best = max(self.last_detections, key=lambda item: item["confidence"], default=None)
        class_name = best["class_name"] if best else "not_detected"
        translated = {
            "immature": "Immature cataract",
            "mature": "Mature cataract",
            "normal": "Normal",
        }.get(class_name.lower(), "Not detected")
        payload = {
            "patient": self.patient_info,
            "prediction": {
                "class_name": class_name,
                "label": translated,
                "confidence": float(best["confidence"]) if best else 0,
                "object_count": len(self.last_detections),
            },
            "image": base64.b64encode(encoded.tobytes()).decode("ascii"),
        }
        server_url = os.environ.get("IRISLENS_SERVER_URL", "http://127.0.0.1:8000").rstrip("/")
        request = urllib.request.Request(
            f"{server_url}/api/examinations",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                result = json.loads(response.read().decode("utf-8"))
            QMessageBox.information(
                self,
                "Sent to Doctor Portal",
                f"Examination results for {self.patient_info.get('name', '—')} were sent successfully.\n\n"
                f"Prediction: {self.last_prediction}\n"
                f"Doctor portal: {server_url}\nExamination ID: {result.get('id', '—')}",
            )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            QMessageBox.critical(
                self,
                "Delivery Failed",
                f"Could not connect to the doctor portal at {server_url}.\n\n"
                "Make sure web_app.py is running, then try sending again.\n\n"
                f"Details: {exc}",
            )

    def update_video_frame(self):
        if not self.is_streaming:
            return

        frame = None
        if self.cap and self.cap.isOpened():
            ok, camera_frame = self.cap.read()
            if ok:
                frame = camera_frame

        self.current_frame_bgr = frame.copy() if frame is not None else None
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "CAMERA NOT DETECTED", (75, 235), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)
            self.ai_badge.hide()
        else:
            self.ai_badge.setText("WEBCAM ACTIVE - YOLO26 MODEL")
            self.ai_badge.show()
        self.video_label.setPixmap(QPixmap.fromImage(self._frame_to_qimage(frame)).scaled(
            self.video_label.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
        ))

    @staticmethod
    def _frame_to_qimage(frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        return QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()

    def on_back_clicked(self):
        self.back_to_registration.emit()

    def closeEvent(self, event):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        event.accept()


class MainWindow(QMainWindow):
    """Main window with a QStackedWidget for navigating three pages."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IrisLens AI - Slit Lamp Cataract Screening")
        self.resize(1024, 560)
        self.setMinimumSize(1024, 560)
        
        self.stacked_widget = QStackedWidget(self)
        self.setCentralWidget(self.stacked_widget)
        
        # 1. Page 1: Splash Screen
        self.splash_screen = SplashScreen(self)
        self.splash_screen.navigate_to_registration.connect(self.go_to_registration)
        self.stacked_widget.addWidget(self.splash_screen)
        
        # 2. Page 2: New Patient Registration
        self.reg_screen = RegistrationScreen(self)
        self.reg_screen.navigate_to_examination.connect(self.go_to_examination)
        self.stacked_widget.addWidget(self.reg_screen)
        
        # Create the examination screen only when needed. It scans cameras and
        # loads the NCNN model, which can otherwise block the splash screen
        # before Qt has had a chance to display it.
        self.exam_screen = None
        
        # Start on the splash screen
        self.stacked_widget.setCurrentIndex(0)

    def go_to_registration(self):
        self.stacked_widget.setCurrentIndex(1)

    def go_to_examination(self, patient_data):
        if self.exam_screen is None:
            self.exam_screen = ExaminationScreen(self)
            self.exam_screen.back_to_registration.connect(self.go_to_registration)
            self.stacked_widget.addWidget(self.exam_screen)
        self.exam_screen.set_patient_data(patient_data)
        self.stacked_widget.setCurrentIndex(2)


def main():
    # Enable High-DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Set default app font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
