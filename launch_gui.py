import os
import sys
import traceback
import io
import re
import math
import pyperclip
import datetime
import markdown2
import requests
import torch
import time
import torchaudio
import json
import pyglet
import sounddevice as sd
from pypdf import PdfReader
from io import BytesIO
from transformers import pipeline
from transformers import AutoTokenizer
from PyQt5.QtWidgets import QApplication, QAbstractItemView, QShortcut, QMainWindow, QDialog, QVBoxLayout, QTextEdit, QLineEdit, QPushButton, QLabel, QWidget, QCheckBox, QListWidget, QListWidgetItem, QSizePolicy, QInputDialog, QHBoxLayout, QComboBox, QMessageBox, QTextEdit
from PyQt5.QtGui import QPixmap, QMouseEvent, QKeySequence, QImage
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QSize
from PyQt5.QtGui import QKeySequence
from pydub import AudioSegment
from pygments import highlight
from pygments.lexers import guess_lexer
import qdarkstyle
from PyQt5.QtWebEngineWidgets import QWebEngineView
from pygments.formatters import HtmlFormatter

from .assistant import OpenAIAssistant, LocalAssistant
from .tts_elevenlabs import TTSElevenlabs
from .tools import WebSearch, WolframAlpha
from .prompt_manager import PromptManager
from .conversation_manager import ConversationManager
from .memory_manager import MemoryManager


class HotkeyThread(QThread):
    hotkeyActivated = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self):
        shortcut = QShortcut(QKeySequence("Ctrl+Alt+Shift+Z"), self.parent().parent())
        shortcut.activated.connect(self.hotkeyActivated.emit)


class FloatingIcon(QMainWindow):
    def __init__(self, chat_config=None, text2audio_api_key=None, text2audio_voice='Jarvis', wolfram_app_id=None, mode='openai', parent=None):
        super().__init__(parent)
        self.initUI()
        self.setWindowFlag(Qt.WindowStaysOnTopHint)
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowOpacity(0.9)
        self.setFixedSize(96, 96)
        self.mousePressPos = None
        self.mouseMovePos = None
        if mode == 'openai':
            self.chat_dialog = ChatDialog(text2audio=TTSElevenlabs(api_key=text2audio_api_key) if text2audio_api_key != None else None, text2audio_voice='Jarvis' if text2audio_voice is None else text2audio_voice, assistant=OpenAIAssistant(memory_manager=MemoryManager(), **chat_config), wolfram_app_id=wolfram_app_id, mode='openai')
        elif mode == 'local':
            self.chat_dialog = ChatDialog(text2audio=TTSElevenlabs(api_key=text2audio_api_key) if text2audio_api_key != None else None, text2audio_voice='Jarvis' if text2audio_voice is None else text2audio_voice, assistant=LocalAssistant(memory_manager=MemoryManager(), **chat_config), wolfram_app_id=wolfram_app_id, mode='local')
        self.chat_width = 800
        self.chat_height = 800
        self.chat_x, self.chat_y = int(QApplication.instance().desktop().screenGeometry().center().x() - self.chat_width / 2), int(QApplication.instance().desktop().screenGeometry().center().y() - self.chat_height / 2)
        self.screen_geometry = QApplication.instance().desktop().screenGeometry()

        # Set up the hotkey
        self.hotkeyThread = HotkeyThread(self)
        self.hotkeyThread.hotkeyActivated.connect(self.showChatDialog)
        self.hotkeyThread.start()

    def initUI(self):
        label = QLabel()
        image = QImage("logo.png")
        image = image.scaled(QSize(96, 96), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pixmap = QPixmap.fromImage(image)
        label.setPixmap(pixmap)
        self.setCentralWidget(label)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.mouse_press_pos = event.globalPos()
            self.mouse_move_pos = event.globalPos()

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() == Qt.LeftButton:
            curr_pos = event.globalPos()
            diff = curr_pos - self.mouse_move_pos
            new_pos = self.pos() + diff
            if new_pos.x() < self.screen_geometry.left() or new_pos.x() + self.width() > self.screen_geometry.right():
                new_pos.setX(min(max(new_pos.x(), self.screen_geometry.left()), self.screen_geometry.right() - self.width()))
            if new_pos.y() < self.screen_geometry.top() or new_pos.y() + self.height() > self.screen_geometry.bottom():
                new_pos.setY(min(max(new_pos.y(), self.screen_geometry.top()), self.screen_geometry.bottom() - self.height()))
            self.move(new_pos)
            self.mouse_move_pos = curr_pos

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            curr_pos = event.globalPos()
            diff = curr_pos - self.mouse_press_pos
            if diff.manhattanLength() > 3:
                return
            self.showChatDialog()

    def showChatDialog(self):
        if self.chat_dialog.isVisible():
            self.chat_width, self.chat_height = self.chat_dialog.width(), self.chat_dialog.height()
            self.chat_x, self.chat_y = self.chat_dialog.x(), self.chat_dialog.y()
            self.chat_dialog.hide()
        else:
            self.chat_dialog.resize(self.chat_width, self.chat_height)
            self.chat_dialog.move(self.chat_x, self.chat_y)
            self.chat_dialog.show()


class MessageItem(QWidget):
    def __init__(self, message, conversation_id, parent_id, is_user=True, parent=None):
        super().__init__(parent)
        self.is_user = is_user
        self.conversation_id = conversation_id
        self.parent_id = parent_id

        # Set style sheet
        self.setStyleSheet(qdarkstyle.load_stylesheet())

        layout = QVBoxLayout()

        if self.is_user:
            self.message = message[5:]
        else:
            self.message = message[11:]

        rendered_message = self.render_message(message)

        layout.addWidget(rendered_message)

        if not self.is_user and os.path.isfile(f"generated_audio/speech_{self.conversation_id}_{self.parent_id}.wav"):
            play_button = QPushButton("Play")
            play_button.setStyleSheet("width: 50px; height: 30px;")
            layout.addWidget(play_button)
            play_button.clicked.connect(self.play_audio)

        # set height to content height
        layout.addStretch(1)
        self.setLayout(layout)

    def mousePressEvent(self, event):
        pyperclip.copy(self.message)

    def play_audio(self):
        music = pyglet.media.load(f"generated_audio/speech_{self.conversation_id}_{self.parent_id}.wav", streaming=False)
        music.play()

    def render_message(self, message):
        html_parts = []
        code_blocks = re.split(r'(```(.*?)```)', message, flags=re.DOTALL)
        for i, part in enumerate(code_blocks):
            if i % 3 == 0:  # This is a non-code part
                html_parts.append(markdown2.markdown(part.strip(), extras=["fenced-code-blocks"]))
            elif i % 3 == 2:  # This is a code part
                lexer = guess_lexer(part)
                formatter = HtmlFormatter()
                highlighted_code = highlight(part, lexer, formatter)

                # Add a bar above the code block and wrap both in a single div
                code_block_header = f"""
                <div class="code-block-container">
                    <div class="code-block-header">
                        <span class="language">{lexer.name}</span>
                        <button class="copy-button" onclick="copyToClipboard({i // 3})">Copy to clipboard</button>
                    </div>
                """
                html_parts.append(code_block_header + highlighted_code + '</div>')

        # Include CSS styles for syntax highlighting and the code block header
        formatter = HtmlFormatter()
        css_styles = formatter.get_style_defs('.highlight')
        header_styles = """
        .code-block-container {
            border-radius: 4px;
            overflow: hidden;
        }
        .code-block-header {
            display: flex;
            justify-content: space-between;
            background-color: #f0f0f0;
            padding: 4px 8px;
            font-family: monospace;
            font-size: 0.8em;
        }
        .language {
            color: #333;
        }
        .copy-button {
            border: none;
            background-color: #f0f0f0;
            color: #007acc;
            cursor: pointer;
            text-decoration: underline;
        }
        .copy-button:hover {
            color: #005999;
        }
        """
        # Add chat bubble CSS styles
        chat_bubble_styles = f"""
        html {{
            height: 100%;
        }}
        body {{
            background-color: transparent;
            display: flex;
            flex-direction: column;
            align-items: {'flex-end' if self.is_user else 'flex-start'};
        }}
        .chat-bubble {{
            background-color: {'#9b30ff' if self.is_user else '#4d94ff'};
            color: white;
            border-radius: 16px;
            padding: 8px 16px;
            margin-bottom: 8px;
            display: inline-block;
            max-width: 80%;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
        }}
        """
        final_html = f"<style>{css_styles}{header_styles}{chat_bubble_styles}</style><div class='chat-bubble'>" + ''.join(html_parts) + "</div>"

        # Add JavaScript for "copy to clipboard" functionality
        js_script = """
        <script>
            function copyToClipboard(blockIndex) {
                const codeBlocks = document.querySelectorAll('.highlight > pre');
                if (blockIndex < codeBlocks.length) {
                    const codeBlock = codeBlocks[blockIndex];
                    const range = document.createRange();
                    range.selectNodeContents(codeBlock);
                    const selection = window.getSelection();
                    selection.removeAllRanges();
                    selection.addRange(range);
                    document.execCommand('copy');
                    selection.removeAllRanges();
                }
            }
        </script>
        """
        final_html += js_script

        web_view = QWebEngineView()
        web_view.setHtml(final_html)

        # Set the background color of the QWebEngineView to transparent
        web_view.setAttribute(Qt.WA_TranslucentBackground, True)
        web_view.setAutoFillBackground(False)
        web_view.page().setBackgroundColor(Qt.transparent)

        return web_view

class ManageConversationsDialog(QDialog):
    def __init__(self, cm, parent=None):
        super().__init__(parent)
        self.cm = cm
        self.conversation_id = None
        self.conversations = None
        self.initUI()
        self.setWindowFlag(Qt.WindowStaysOnTopHint)

    def initUI(self):
        layout = QVBoxLayout()

        self.conversations_list = QListWidget()
        layout.addWidget(self.conversations_list)

        self.populate_conversations_list()

        delete_button = QPushButton("Delete")
        layout.addWidget(delete_button)
        delete_button.clicked.connect(self.delete_conversation)

        delete_all_button = QPushButton("Delete All")
        layout.addWidget(delete_all_button)
        delete_all_button.clicked.connect(self.delete_all_conversation)

        rename_button = QPushButton("Rename")
        layout.addWidget(rename_button)
        rename_button.clicked.connect(self.rename_conversation)

        refresh_button = QPushButton("Refresh")
        layout.addWidget(refresh_button)
        refresh_button.clicked.connect(self.refresh_conversations)

        change_button = QPushButton("Change")
        layout.addWidget(change_button)
        change_button.clicked.connect(self.change_conversation)

        self.setLayout(layout)

        # Set style sheet
        self.setStyleSheet(qdarkstyle.load_stylesheet())

    def populate_conversations_list(self, refresh=False):
        if refresh or self.conversations is None:
            self.conversations = self.cm.get_conversations()
            for conversation in self.conversations:
                self.conversations_list.addItem(conversation['title'])

    def change_conversation(self):
        index = self.conversations_list.currentRow()
        self.conversation_id = self.conversations[index]['id']
        self.accept()

    def delete_conversation(self):
        # confirm delete
        confirm = QMessageBox.question(self, "Delete Conversation", "Are you sure you want to delete this conversation?", QMessageBox.Yes | QMessageBox.No)
        if confirm == QMessageBox.No:
            return
        index = self.conversations_list.currentRow()
        self.conversations_list.takeItem(index)
        conversation = self.conversations.pop(index)
        self.cm.delete_conversation(conversation_id=conversation['id'])

    def delete_all_conversation(self):
        # confirm delete
        confirm = QMessageBox.question(self, "Delete All Conversations", "Are you sure you want to delete all conversations?", QMessageBox.Yes | QMessageBox.No)
        if confirm == QMessageBox.No:
            return
        self.conversations_list.clear()
        self.cm.delete_all_conversations()
        self.conversations = []

    def rename_conversation(self):
        index = self.conversations_list.currentRow()
        conversation = self.conversations[index]
        new_name, ok = QInputDialog.getText(self, "Rename Conversation", "Enter new name for conversation:", QLineEdit.Normal, conversation['title'])
        if ok:
            self.conversations_list.item(index).setText(new_name)
            self.conversations[index]['title'] = new_name
            self.cm.rename_conversation(title=new_name, conversation_id=conversation['id'])

    def refresh_conversations(self):
        self.conversations_list.clear()
        self.populate_conversations_list(refresh=True)


class ManagePromptsDialog(QDialog):
    def __init__(self, pm, parent=None):
        super().__init__(parent)
        self.pm = pm
        self.prompts = []
        self.tags = None
        self.selected_tags = []
        self.tag_mode = 'any'
        self.prepend_first_message = True
        self.prepend = False
        self.initUI()
        self.setWindowFlag(Qt.WindowStaysOnTopHint)

    def initUI(self):
        layout = QVBoxLayout()

        self.prompt_list = QListWidget()
        layout.addWidget(self.prompt_list)

        self.selected_tags_layout = QHBoxLayout()
        layout.addLayout(self.selected_tags_layout)

        # add pagination
        pagination_layout = QHBoxLayout()
        self.page = 1
        self.page_size = 20
        self.total_pages = 1
        self.total_prompts = len(self.prompts)
        self.page_label = QLabel(f"Page {self.page} of {self.total_pages}")
        pagination_layout.addWidget(self.page_label)
        self.first_page_button = QPushButton("First")
        pagination_layout.addWidget(self.first_page_button)
        self.first_page_button.clicked.connect(self.first_page)
        self.prev_page_button = QPushButton("Prev")
        pagination_layout.addWidget(self.prev_page_button)
        self.prev_page_button.clicked.connect(self.prev_page)
        self.next_page_button = QPushButton("Next")
        pagination_layout.addWidget(self.next_page_button)
        self.next_page_button.clicked.connect(self.next_page)
        self.last_page_button = QPushButton("Last")
        pagination_layout.addWidget(self.last_page_button)
        self.last_page_button.clicked.connect(self.last_page)
        layout.addLayout(pagination_layout)

        # 2 dropdowns in a row
        row = QHBoxLayout()
        self.tag_mode_dropdown = QComboBox(self)
        self.tag_mode_dropdown.addItems(["any", "all"])
        self.tag_mode_dropdown.currentIndexChanged.connect(self.handle_tag_mode_change)
        row.addWidget(self.tag_mode_dropdown)
        self.tag_dropdown = QComboBox(self)
        self.tag_dropdown.setEditable(True)
        row.addWidget(self.tag_dropdown)
        self.tag_dropdown.view().pressed.connect(self.handle_tag_dropdown_change_with_index)
        self.tag_dropdown.lineEdit().returnPressed.connect(self.handle_tag_dropdown_change)
        layout.addLayout(row)

        self.populate_tags()
        self.populate_prompt_list(refresh=True)

        delete_button = QPushButton("Delete")
        layout.addWidget(delete_button)
        delete_button.clicked.connect(self.delete_prompt)

        create_button = QPushButton("Create")
        layout.addWidget(create_button)
        create_button.clicked.connect(self.create_prompt)

        update_button = QPushButton("Update")
        layout.addWidget(update_button)
        update_button.clicked.connect(self.update_prompt)

        select_button = QPushButton("Select")
        layout.addWidget(select_button)
        select_button.clicked.connect(self.select_prompt)
        
        self.prepend_first_message_checkbox = QCheckBox("Prepend to First Message")
        layout.addWidget(self.prepend_first_message_checkbox)
        self.prepend_first_message_checkbox.stateChanged.connect(self.handle_prepend_first_message_checkbox_state_change)
        self.prepend_first_message_checkbox.setChecked(True)

        self.prepend_checkbox = QCheckBox("Prepend to All Messages")
        layout.addWidget(self.prepend_checkbox)
        self.prepend_checkbox.stateChanged.connect(self.handle_prepend_checkbox_state_change)
        self.prepend_checkbox.setChecked(False)

        self.setLayout(layout)

        # Set style sheet
        self.setStyleSheet(qdarkstyle.load_stylesheet())

    def first_page(self):
        self.page = 1
        self.update_page()

    def prev_page(self):
        if self.page <= 1:
            return
        self.page -= 1
        self.update_page()

    def next_page(self):
        if self.page >= self.total_pages:
            return
        self.page += 1
        self.update_page()

    def last_page(self):
        self.page = self.total_pages
        self.update_page()

    def update_page(self):
        self.populate_prompt_list(tags=self.selected_tags, refresh=True)

    def update_selected_tags_display(self):
        # Remove any existing tag buttons
        for i in reversed(range(self.selected_tags_layout.count())):
            widget = self.selected_tags_layout.itemAt(i).widget()
            widget.setParent(None)
        # Add a button for each selected tag
        for tag in self.selected_tags:
            button = QPushButton(tag)
            button.clicked.connect(lambda checked, tag=tag: self.remove_tag(tag))
            self.selected_tags_layout.addWidget(button)

    def handle_tag_dropdown_change(self):
        item = self.tag_dropdown.currentText()
        if item not in self.tags or item in self.selected_tags:
            return
        self.selected_tags.append(item)
        # sort tags
        self.selected_tags.sort()
        self.update_selected_tags_display()
        self.populate_prompt_list(tags=self.selected_tags, refresh=True)

    def handle_tag_dropdown_change_with_index(self, index):
        item = self.tag_dropdown.itemText(index.row())
        if item not in self.tags or item in self.selected_tags:
            return
        self.selected_tags.append(item)
        # sort tags
        self.selected_tags.sort()
        self.update_selected_tags_display()
        self.populate_prompt_list(tags=self.selected_tags, refresh=True)

    def clear_tags(self):
        self.selected_tags = []
        self.tag_dropdown.view().clearSelection()
        self.update_selected_tags_display()
        self.populate_prompt_list(tags=self.selected_tags, refresh=True)

    def remove_tag(self, tag):
        self.selected_tags.remove(tag)
        self.update_selected_tags_display()
        self.populate_prompt_list(tags=self.selected_tags, refresh=True)

    def handle_prepend_first_message_checkbox_state_change(self, state):
        if state == Qt.Checked:
            self.prepend_first_message = True
            self.prepend_first_message_checkbox.setChecked(True) 
        else:
            self.prepend_first_message = False
            self.prepend_first_message_checkbox.setChecked(False)

    def handle_prepend_checkbox_state_change(self, state):
        if state == Qt.Checked:
            self.prepend = True
            self.prepend_checkbox.setChecked(True) 
        else:
            self.prepend = False
            self.prepend_checkbox.setChecked(False)

    def handle_tag_mode_change(self, index):
        self.tag_mode = self.tag_mode_dropdown.itemText(index)
        self.populate_prompt_list(tags=self.selected_tags, refresh=True)

    def populate_tags(self, refresh=False):
        if refresh or self.tags is None:
            self.tags = self.pm.get_tags()

    def populate_prompt_list(self, tags=None, refresh=False):
        if refresh or self.prompts is None:
            self.prompt_list.clear()
            self.prompts = self.pm.get_prompts(tags=tags, tag_filter_mode=self.tag_mode, offset=(self.page-1)*self.page_size, limit=self.page_size)
            self.total_prompts = self.pm.get_prompts_count(tags=tags, tag_filter_mode=self.tag_mode)
            self.total_pages = math.ceil(self.total_prompts / self.page_size)
            self.page_label.setText(f"Page {self.page} of {self.total_pages}")
            for prompt in self.prompts:
                self.prompt_list.addItem(prompt['title'])

    def populate_tags(self, refresh=False):
        if refresh or self.tags is None:
            self.tag_dropdown.clear()
            self.tags = self.pm.get_tags()
            for tag in self.tags:
                self.tag_dropdown.addItem(tag)

    def update_prompt(self):
        index = self.prompt_list.currentRow()
        prompt = self.prompts[index]
        # Get new title, prompt, tags, and variables (prepopulate with old values)
        new_title, ok = QInputDialog.getText(self, "Update Prompt", "Enter new title for prompt:", QLineEdit.Normal, prompt['title'])
        if ok:
            new_prompt, ok = QInputDialog.getText(self, "Update Prompt", "Enter new prompt:", QLineEdit.Normal, prompt['prompt'])
            if ok:
                new_variables, ok = QInputDialog.getText(self, "Update Prompt", "Enter new variables (comma separated):", QLineEdit.Normal, ", ".join([var.strip() for var in prompt['variables']]))
                if ok:
                    new_tags, ok = QInputDialog.getText(self, "Update Prompt", "Enter new tags (comma separated):", QLineEdit.Normal, ", ".join([tag.strip() for tag in prompt['tags']]))
                    if ok:
                        # get notes
                        new_notes, ok = QInputDialog.getText(self, "Update Prompt", "Enter new notes:", QLineEdit.Normal, prompt['notes'])
                        if ok:
                            self.pm.update_prompt(prompt_id=prompt['id'], title=new_title, prompt=new_prompt, tags=new_tags.split(', '), variables=new_variables.split(', '), notes=new_notes)
                            self.refresh_prompts(refresh=True)

    def delete_prompt(self):
        # confirm delete
        reply = QMessageBox.question(self, 'Delete Prompt', 'Are you sure you want to delete this prompt?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.No:
            return
        index = self.prompt_list.currentRow()
        self.prompt_list.takeItem(index)
        prompt = self.prompts.pop(index)
        self.pm.delete_prompt(prompt_id=prompt['id'])

    def create_prompt(self):
        title, ok = QInputDialog.getText(self, "Create Prompt", "Enter title for prompt:")
        if ok:
            prompt, ok = QInputDialog.getText(self, "Create Prompt", "Enter prompt:")
            if ok:
                variables, ok = QInputDialog.getText(self, "Create Prompt", "Enter variables (comma separated):")
                if ok:
                    tags, ok = QInputDialog.getText(self, "Create Prompt", "Enter tags (comma separated):")
                    if ok:
                        notes, ok = QInputDialog.getText(self, "Create Prompt", "Enter notes:")
                        if ok:
                            prompt_id = self.pm.add_prompt(title=title, prompt=prompt, tags=tags.split(', '), variables=variables.split(', '), notes=notes)
                            self.prompts.append({'id': prompt_id, 'title': title, 'prompt': prompt, 'tags': tags.split(', '), 'variables': variables.split(', '), 'notes': notes})
                            self.refresh_prompts(refresh=True)

    def select_prompt(self):
        index = self.prompt_list.currentRow()
        self.prompt_id = self.prompts[index]['id']
        self.accept()

    def refresh_prompts(self):
        self.prompt_list.clear()
        self.populate_prompt_list(refresh=True)


class ChatDialog(QDialog):
    def __init__(self, text2audio=None, text2audio_voice='Jarvis', assistant=None, device="cuda:0" if torch.cuda.is_available() else "cpu", audio2text="openai/whisper-large-v2", wolfram_app_id=None, prompt_database=None, conversation_database=None, mode='openai'):
        super().__init__()
        self.continue_text = 'Continue writing please'
        self.web_search_and_summarize_text = """Web search results:

WEB_RESULTS*
Current date: CURRENT_DATE*

Instructions: Using the provided web search results, write a comprehensive reply to the given query. Make sure to cite results using [[number](URL)] notation after the reference. If the provided search results refer to multiple subjects with the same name, write separate answers for each subject.
Query: QUERY*"""
        self.web_search_and_summarize_no_cite_text = """Web search results:

WEB_RESULTS*
Current date: CURRENT_DATE*

Instructions: Using the provided web search results, write a comprehensive reply to the given query. If the provided search results refer to multiple subjects with the same name, write separate answers for each subject.
Query: QUERY*"""
        self.summarize_text = """Summarize part CURRENT_PAGE* out of NUM_PAGES*. Stay under PAGE_SUMMARY_WORD_LIMIT* words. Only summarize the new text. Do not repeat what is in the previous summary, use it for context only.

Previous Summary: PREVIOUS_SUMMARY*

Text: PAGE_TEXT*"""
        self.final_summarize_text = """Summarize the following short summaries. Stay under PAGE_SUMMARY_WORD_LIMIT* words.

Summaries: SUMMARIES*"""
        self.text2audio = text2audio
        self.web = WebSearch()
        self.wolfram = None
        if wolfram_app_id is not None:
            self.wolfram = WolframAlpha(app_id=wolfram_app_id)
        self.assistant = assistant
        self.audio2text_pipe = pipeline(
            "automatic-speech-recognition",
            model=audio2text,
            chunk_length_s=30,
            device=device,
        )
        if self.text2audio is None:
            self.text2speech_enabled = False
        else:
            try:
                self.text2audio.set_active_voice(voice_name=text2audio_voice)
                self.text2speech_enabled = True
            except:
                self.text2speech_enabled = False
        self.pm = PromptManager(database=prompt_database)
        self.cm = ConversationManager(database=conversation_database)
        self.prompt = None
        self.prepend_first_prompt = False
        self.prepend_prompt = False
        self.tokenizer = AutoTokenizer.from_pretrained("GPT2-large")
        self.conversation_id = None
        self.parent_id = None
        self.format_cache = {}
        self.mode = mode
        self.initUI()
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
    
    def initUI(self):
        # Set up the layout for the chat dialog
        layout = QVBoxLayout()
        
        # Add a QListWidget widget to display previous conversations
        self.conversations_list = QListWidget()
        self.conversations_list.setStyleSheet("background-color: transparent;")
        self.conversations_list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        layout.addWidget(self.conversations_list)
        
        # Add a QTextEdit widget to allow the user to type and send messages
        self.message_edit = QTextEdit()
        self.message_edit.setPlaceholderText("Prompt")
        layout.addWidget(self.message_edit)
        self.message_edit.text = lambda : self.message_edit.toPlainText()
        self.message_edit.insert = lambda text: self.message_edit.insertPlainText(text)
        self.message_edit.keyPressEvent = self.keyPressEvent
        self.message_edit.setFixedHeight(100)
        self.message_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Add 2 widgets in one row
        widget_row = QHBoxLayout()
        layout.addLayout(widget_row)
        # Add a QLabel widget to display the character count
        self.char_count_label = QLabel("0/4096")
        widget_row.addWidget(self.char_count_label)

        if self.text2audio is not None:
            # Add a QCheckBox widget to control text-to-speech
            self.text2speech_checkbox = QCheckBox("Text-to-Speech")
            widget_row.addWidget(self.text2speech_checkbox)
            widget_row.addStretch()
            self.text2speech_checkbox.stateChanged.connect(self.handle_text2speech_checkbox_state_change)
            # Set the checkbox to checked by default
            if self.text2speech_enabled:
                self.text2speech_checkbox.setChecked(True)
            else:
                self.text2speech_checkbox.setChecked(False)

        # Add 3 buttons in one row
        button_row = QHBoxLayout()
        layout.addLayout(button_row)
        # Add a QPushButton to send the message
        send_button = QPushButton("Send")
        button_row.addWidget(send_button)
        send_button.clicked.connect(self.sendMessage)
        # Add a "Record" button to record audio
        record_button = QPushButton("Record")
        button_row.addWidget(record_button)
        record_button.clicked.connect(self.recordMessage)
        # Add a QPushButton to continue the message
        continue_button = QPushButton("Continue")
        button_row.addWidget(continue_button)
        continue_button.clicked.connect(self.continueMessage)

        # Add 5 buttons in one row
        format_button_row = QHBoxLayout()
        layout.addLayout(format_button_row)
        # Add a QPushButton to insert web_search query
        web_search_button = QPushButton("Web Search")
        format_button_row.addWidget(web_search_button)
        web_search_button.clicked.connect(self.insert_web_search_query)
        # Add a QPushButton to insert web_search_and_summarize query
        web_search_and_summarize_button = QPushButton("Web Search and Summarize")
        format_button_row.addWidget(web_search_and_summarize_button)
        web_search_and_summarize_button.clicked.connect(self.insert_web_search_and_summarize_query)
        # Add a QPushButton to insert pdf_summarize query
        pdf_summarize_button = QPushButton("PDF Summarize")
        format_button_row.addWidget(pdf_summarize_button)
        pdf_summarize_button.clicked.connect(self.insert_summarize_pdf)
        # Add a QPushButton to insert Wolfram query
        if self.wolfram is not None:
            wolfram_button = QPushButton("Wolfram")
            format_button_row.addWidget(wolfram_button)
            wolfram_button.clicked.connect(self.insert_wolfram_query)
        # Add a QPushButton to insert date query
        date_button = QPushButton("Date")
        format_button_row.addWidget(date_button)
        date_button.clicked.connect(self.insert_date_query)
        
        # Add 2 buttons in one row
        format_button_row2 = QHBoxLayout()
        layout.addLayout(format_button_row2)
        # Add a QPushButton to format text
        format_button = QPushButton("Format")
        format_button_row2.addWidget(format_button)
        format_button.clicked.connect(self.format_text)
        # Add a QPushButton to unformat text
        unformat_button = QPushButton("Unformat")
        format_button_row2.addWidget(unformat_button)
        unformat_button.clicked.connect(self.unformat_text)

        # Add 3 buttons in one row
        edit_prompt_row = QHBoxLayout()
        # Add a QPushButton to edit the continue message
        edit_continue_button = QPushButton("Edit Continue Prompt")
        edit_prompt_row.addWidget(edit_continue_button)
        edit_continue_button.clicked.connect(self.edit_continue_prompt)
        # Add a QPushButton to edit the web search summarize prompt
        edit_web_search_and_summarize_button = QPushButton("Edit Web Search Summary Prompt")
        edit_prompt_row.addWidget(edit_web_search_and_summarize_button)
        edit_web_search_and_summarize_button.clicked.connect(self.edit_web_search_and_summarize_prompt)
        # Add a QPushButton to edit the web search summarize no citation prompt
        edit_web_search_and_summarize_no_citation_button = QPushButton("Edit Web Search Summary No Citation Prompt")
        edit_prompt_row.addWidget(edit_web_search_and_summarize_no_citation_button)
        edit_web_search_and_summarize_no_citation_button.clicked.connect(self.edit_web_search_and_summarize_prompt_no_cite)
        layout.addLayout(edit_prompt_row)

        # Add 2 buttons in one row
        manage_row = QHBoxLayout()
        layout.addLayout(manage_row)
        # Add a QPushButton to manage conversations
        manage_conversation_button = QPushButton("Manage Conversations")
        manage_row.addWidget(manage_conversation_button)
        manage_conversation_button.clicked.connect(self.conversationManager)
        # Add a QPushButton to manage prompts
        manage_prompt_button = QPushButton("Manage Prompts")
        manage_row.addWidget(manage_prompt_button)
        manage_prompt_button.clicked.connect(self.promptManager)

        # Add a QPushButton to start a new conversation
        new_conversation_button = QPushButton("New Conversation")
        layout.addWidget(new_conversation_button)
        new_conversation_button.clicked.connect(self.newConversation)
        
        # Set the layout for the chat dialog
        self.setLayout(layout)

        # Set style sheet
        self.setStyleSheet(qdarkstyle.load_stylesheet())

        # Connect the textChanged signal of the QLineEdit widget to a slot that updates the character count label
        self.message_edit.textChanged.connect(self.updateCharCount)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return and event.modifiers() == Qt.ShiftModifier:
            self.message_edit.insertPlainText('\n')
        elif event.key() == Qt.Key_Return:
            self.sendMessage()
        else:
            QTextEdit.keyPressEvent(self.message_edit, event)

    def newConversation(self):
        # Make a new conversation
        self.conversation_id = None
        self.parent_id = None
        self.conversations_list.clear()
        self.assistant.short_term_memory = []

    def insert_web_search_query(self):
        # Insert a web_search query at the cursor position
        self.message_edit.insert("{web_search(query=, region=us-en, safesearch=Off, time=y, max_results=20, page=1, output=None, download=False)}")

    def insert_wolfram_query(self):
        # Insert a wolfram query at the cursor position
        self.message_edit.insert("{wolfram_short_answer(query=)}")

    def insert_web_search_and_summarize_query(self):
        # Insert a web_search_and_summarize query at the cursor position
        self.message_edit.insert("{web_search_and_summarize(query=, region=us-en, safesearch=Off, time=y, max_results=20, page=1, output=None, download=False, citations=False)}")

    def insert_date_query(self):
        # Insert a date query at the cursor position
        self.message_edit.insert("{date(format=%Y-%m-%d %H:%M:%S)}")

    def insert_summarize_pdf(self):
        # Insert a summarize_pdf query at the cursor position
        self.message_edit.insert("{summarize_pdf(url=, target_words=1000, target_words_per_page=500)}")

    def summarize_pdf(self, url, target_words=1000, target_words_per_page=500, num_attempts=3, render=False):
        # Summarize a pdf
        reader = PdfReader(BytesIO(requests.get(url).content))
        texts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if len(texts) > 0:
                token_count = self.tokenizer(texts[-1] + page_text, return_length=True)['length'][0]
                if token_count < 3200:
                    texts[-1] += page_text
                else:
                    texts.append(page_text)
            else:
                texts.append(page_text)
        
        summaries = []
        for i, text in enumerate(texts):
            try:
                if i > 0:
                    time.sleep(3)
                prompt = self.pm.replace_prompt_variables(self.summarize_text, {
                    'current_page': i+1,
                    'num_pages': len(texts),
                    'page_summary_word_limit': target_words_per_page,
                    'previous_summary': '' if len(summaries) == 0 else summaries[-1],
                    'page_text': text
                })
                if render:
                    self.add_message("You: " + prompt, is_user=True)
                    self.cm.add_message(prompt, 'user', mode=self.mode)
                    # Show a loading message in the conversation list
                    self.add_message("Assistant: Loading...", is_user=False)
                    QApplication.processEvents()
                # Send the message to the assistant and get the response
                if self.mode == 'openai':
                    response = self.assistant.get_chat_response(prompt=prompt).choices[0].message.content
                else:
                    response = self.assistant.get_chat_response(prompt=prompt)['content']
                if render:
                    # Remove the previous message
                    self.conversations_list.takeItem(self.conversations_list.count()-1)
                    self.add_message("Assistant: " + response, is_user=False)
                    self.cm.add_message(response, 'assistant')
                QApplication.processEvents()
                summaries.append(response.strip())
                conversation_id = self.assistant.conversation_id
                parent_id = self.assistant.parent_id
            except:
                time.sleep(3)
                prompt = self.pm.replace_prompt_variables(self.summarize_text, {
                    'current_page': i+1,
                    'num_pages': len(texts),
                    'page_summary_word_limit': target_words_per_page,
                    'previous_summary': '' if len(summaries) == 0 else summaries[-1],
                    'page_text': text
                })
                if render:
                    self.add_message("You: " + prompt, is_user=True)
                    self.cm.add_message(prompt, 'user', mode=self.mode)
                    # Show a loading message in the conversation list
                    self.add_message("Assistant: Loading...", is_user=False)
                    QApplication.processEvents()
                # Send the message to the assistant and get the response
                if self.mode == 'openai':
                    response = self.assistant.get_chat_response(prompt=prompt).choices[0].message.content
                else:
                    response = self.assistant.get_chat_response(prompt=prompt)['content']
                if render:
                    # Remove the previous message
                    self.conversations_list.takeItem(self.conversations_list.count()-1)
                    self.add_message("Assistant: " + response, is_user=False)
                    self.cm.add_message(response, 'assistant')
                QApplication.processEvents()
                summaries.append(response.strip())
                conversation_id = self.assistant.conversation_id
                parent_id = self.assistant.parent_id

        texts = []
        for summary in summaries:
            if len(texts) > 0:
                token_count = self.tokenizer(texts[-1] + '\n' + summary, return_length=True)['length'][0]
                if token_count < 3200:
                    texts[-1] += '\n' + summary
                else:
                    texts.append(summary)
            else:
                texts.append(summary)

        count = 0
        while len(texts) > 1 and count < num_attempts:
            new_texts = []
            for i in range(0, len(texts)):
                try:
                    if i > 0:
                        time.sleep(3)
                    prompt = self.pm.replace_prompt_variables(self.summarize_text, {
                        'current_page': i+1,
                        'num_pages': len(texts),
                        'page_summary_word_limit': target_words_per_page,
                        'previous_summary': '' if len(summaries) == 0 else summaries[-1],
                        'page_text': texts[i]
                    })
                    if render:
                        self.add_message("You: " + prompt, is_user=True)
                        self.cm.add_message(prompt, 'user', mode=self.mode)
                        # Show a loading message in the conversation list
                        self.add_message("Assistant: Loading...", is_user=False)
                        QApplication.processEvents()
                    # Send the message to the assistant and get the response
                    if self.mode == 'openai':
                        response = self.assistant.get_chat_response(prompt=prompt).choices[0].message.content
                    else:
                        response = self.assistant.get_chat_response(prompt=prompt)['content']
                    if render:
                        # Remove the previous message
                        self.conversations_list.takeItem(self.conversations_list.count()-1)
                        self.add_message("Assistant: " + response, is_user=False)
                        self.cm.add_message(response, 'assistant')
                    QApplication.processEvents()
                    new_texts.append(response.strip())
                except:
                    time.sleep(3)
                    prompt = self.pm.replace_prompt_variables(self.summarize_text, {
                        'current_page': i+1,
                        'num_pages': len(texts),
                        'page_summary_word_limit': target_words_per_page,
                        'previous_summary': '' if len(summaries) == 0 else summaries[-1],
                        'page_text': texts[i]
                    })
                    if render:
                        self.add_message("You: " + prompt, is_user=True)
                        self.parent_id = self.cm.add_message(prompt, 'user', mode=self.mode)
                        # Show a loading message in the conversation list
                        self.add_message("Assistant: Loading...", is_user=False)
                        QApplication.processEvents()
                    # Send the message to the assistant and get the response
                    if self.mode == 'openai':
                        response = self.assistant.get_chat_response(prompt=prompt).choices[0].message.content
                    else:
                        response = self.assistant.get_chat_response(prompt=prompt)['content']
                    if render:
                        # Remove the previous message
                        self.conversations_list.takeItem(self.conversations_list.count()-1)
                        self.add_message("Assistant: " + response, is_user=False)
                        self.parent_id = self.cm.add_message(response, 'assistant')
                    QApplication.processEvents()
                    new_texts.append(response.strip())

            texts = []
            for text in new_texts:
                if len(texts) > 0:
                    token_count = self.tokenizer(texts[-1] + '\n' + text, return_length=True)['length'][0]
                    if token_count < 3200:
                        texts[-1] += text
                    else:
                        texts.append(text)
                else:
                    texts.append(text)
            count += 1

        prompt = self.pm.replace_prompt_variables(self.final_summarize_text, {
            'page_summary_word_limit': target_words,
            'summaries': texts[0]
        })

        if render:
            return prompt
        
        if self.mode == 'openai':
            response = self.assistant.get_chat_response(prompt=prompt).choices[0].message.content
        else:
            response = self.assistant.get_chat_response(prompt=prompt)['content']

        summary = response.strip()

        return summary

    def unformat_text(self):
        if self.format_cache == {} or self.format_cache is None:
            return
        # Unformat the text in the message edit
        text = self.message_edit.text()
        # replace all values in the format cache with the query
        for key, value in self.format_cache.items():
            text = text.replace(value, key)
        # set the text in the message edit
        self.message_edit.setText(text)

    def format_text(self, text_=None):
        # Format the text in the message edit
        text = self.message_edit.text()
        if text_ is not None and isinstance(text_, str):
            text = text_
        # find all the web_search queries
        web_search_queries = re.findall(r'{web_search\(query=(.*?)\)}', text)
        # find all the web_search_and_summarize queries
        web_search_and_summarize_queries = re.findall(r'{web_search_and_summarize\(query=(.*?)\)}', text)
        # find all summarize_pdf queries
        pdf_summarize_queries = re.findall(r'{summarize_pdf\(url=(.*?)\)}', text)
        wolfram_queries = []
        if self.wolfram is not None:
            # find all the wolfram queries
            wolfram_queries = re.findall(r'{wolfram_short_answer\(query=(.*?)\)}', text)
        # find all the date queries
        date_queries = re.findall(r'{date\(format=(.*?)\)}', text)
        # replace the queries with the formatted text
        for query_ in web_search_queries:
            if self.format_cache.get("{web_search(query=" + query_ + ")}") is not None:
                web_search = self.format_cache["{web_search(query=" + query_ + ")}"]
                text = text.replace("{web_search(query=" + query_ + ")}", web_search)
                continue
            else:
                # get parameters from the query
                parameters = query_.split(',')
                # get the query
                query = parameters[0].split('=')
                if len(query) > 1:
                    query = query[1].strip()
                else:
                    query = query[0].strip()
                # get the region (may not be present)
                region = 'us-en'
                safe_search = 'Off'
                time = 'y'
                max_results = 20
                page = 1
                output = None
                download = False
                for parameter in parameters:
                    parameter = parameter.strip()
                    if parameter.startswith('region'):
                        region = parameter.split('=')[1].strip()
                    elif parameter.startswith('safesearch'):
                        safe_search = parameter.split('=')[1].strip()
                    elif parameter.startswith('time'):
                        time = parameter.split('=')[1].strip()
                    elif parameter.startswith('max_results'):
                        max_results = int(parameter.split('=')[1].strip())
                    elif parameter.startswith('page'):
                        page = int(parameter.split('=')[1].strip())
                    elif parameter.startswith('output'):
                        output = parameter.split('=')[1].strip()
                    elif parameter.startswith('download'):
                        download = parameter.split('=')[1].strip()
                        if download.lower() == 'true':
                            download = True
                        else:
                            download = False

                # get the web search results
                web_search = self.web.search(keywords=query, region=region, safesearch=safe_search, time=time, max_results=max_results, page=page, output=output, download=download)
                web_search = self.format_web_search_prompt(web_search, return_web_results=True)
                # add the web search to the format cache
                self.format_cache["{web_search(query=" + query_ + ")}"] = web_search
                # replace the query with the web search
                text = text.replace("{web_search(query=" + query_ + ")}", web_search)

        for query_ in web_search_and_summarize_queries:
            if self.format_cache.get("{web_search_and_summarize(query=" + query_ + ")}") is not None:
                web_search_and_summarize = self.format_cache["{web_search_and_summarize(query=" + query_ + ")}"]
                text = text.replace("{web_search_and_summarize(query=" + query_ + ")}", web_search_and_summarize)
                continue
            else:
                # get parameters from the query
                parameters = query_.split(',')
                # get the query
                query = parameters[0].split('=')
                if len(query) > 1:
                    query = query[1].strip()
                else:
                    query = query[0].strip()
                # get the region (may not be present)
                region = 'us-en'
                safe_search = 'Off'
                time = 'y'
                max_results = 20
                page = 1
                output = None
                download = False
                citations = False
                for parameter in parameters:
                    parameter = parameter.strip()
                    if parameter.startswith('region'):
                        region = parameter.split('=')[1].strip()
                    elif parameter.startswith('safesearch'):
                        safe_search = parameter.split('=')[1].strip()
                    elif parameter.startswith('time'):
                        time = parameter.split('=')[1].strip()
                    elif parameter.startswith('max_results'):
                        max_results = int(parameter.split('=')[1].strip())
                    elif parameter.startswith('page'):
                        page = int(parameter.split('=')[1].strip())
                    elif parameter.startswith('output'):
                        output = parameter.split('=')[1].strip()
                    elif parameter.startswith('download'):
                        download = parameter.split('=')[1].strip()
                        if download.lower() == 'true':
                            download = True
                        else:
                            download = False
                    elif parameter.startswith('citations'):
                        citations = parameter.split('=')[1].strip()
                        if citations.lower() == 'true':
                            citations = True
                        else:
                            citations = False

                # get the web search results
                web_search = self.web.search(keywords=query, region=region, safesearch=safe_search, time=time, max_results=max_results, page=page, output=output, download=download)
                # add the web search to the format cache
                web_key = re.sub(r', citations=(True|False)', '', query_)
                self.format_cache["{web_search(query=" + web_key + ")}"] = self.format_web_search_prompt(web_search, return_web_results=True)
                # summarize the web search
                conversation_id = self.conversation_id
                parent_id = self.parent_id
                web_search_and_summarize = self.assistant.get_response(prompt=self.format_web_search_prompt(web_search, query=query, citations=citations), conversation_id=None, parent_id=None)
                response = ''
                for response_ in web_search_and_summarize:
                    response = response_
                if self.mode == 'openai':
                    self.assistant.change_title(title=f'{query} Summarization')
                    self.assistant.reset()
                    self.assistant.conversation_id = conversation_id
                    self.assistant.parent_id = parent_id
                # add the summarized web search to the format cache
                self.format_cache["{web_search_and_summarize(query=" + query_ + ")}"] = response
                # replace the query with the summarized web search
                text = text.replace("{web_search_and_summarize(query=" + query_ + ")}", response)

        for query_ in wolfram_queries:
            if self.format_cache.get("{wolfram_short_answer(query=" + query_ + ")}") is not None:
                wolfram_short_answer = self.format_cache["{wolfram_short_answer(query=" + query_ + ")}"]
                text = text.replace("{wolfram_short_answer(query=" + query_ + ")}", wolfram_short_answer)
                continue
            # get parameters from the query
            parameters = query_.split(',')
            # get the query
            query = parameters[0].split('=')
            if len(query) > 1:
                query = query[1].strip()
            else:
                query = query[0].strip()

            # get the wolfram search results
            wolfram_short_answer = self.wolfram.get_short_answer(query=query)
            answer = f"Query: {wolfram_short_answer['query']}\nAnswer:{wolfram_short_answer['response']}"
            # add the wolfram search to the format cache
            self.format_cache["{wolfram_short_answer(query=" + query_ + ")}"] = answer
            # replace the query with the wolfram search
            text = text.replace("{wolfram_short_answer(query=" + query_ + ")}", answer)

        for query_ in pdf_summarize_queries:
            if self.format_cache.get("{pdf_summarize(url=" + query_ + ")}") is not None:
                pdf_summarize = self.format_cache["{pdf_summarize(url=" + query_ + ")}"]
                text = text.replace("{pdf_summarize(url=" + query_ + ")}", pdf_summarize)
                continue
            # get parameters from the query
            parameters = pdf_summarize_queries[0].split(',')
            # get the url
            url = parameters[0].split('=')
            if len(url) > 1:
                url = url[1].strip()
            else:
                url = url[0].strip()
            # get the params (may not be present)
            target_words = 1000
            target_words_per_page = 500
            for parameter in parameters:
                parameter = parameter.strip()
                if parameter.startswith('target_words') and not parameter.startswith('target_words_'):
                    target_words = int(parameter.split('=')[1].strip())
                elif parameter.startswith('target_words_per_page'):
                    target_words_per_page = int(parameter.split('=')[1].strip())
            # get the pdf summarize results
            pdf_summarize = self.summarize_pdf(url=url, target_words=target_words, target_words_per_page=target_words_per_page)
            # add the pdf summarize to the format cache
            self.format_cache["{pdf_summarize(url=" + query_ + ")}"] = pdf_summarize
            # replace the query with the pdf summarize
            text = text.replace("{pdf_summarize(url=" + query_ + ")}", pdf_summarize)

        for query_ in date_queries:
            # get parameters from the query
            parameters = query_.split(',')
            # get the format
            format = parameters[0].split('=')
            if len(format) > 1:
                format = format[1].strip()
            else:
                format = format[0].strip()
            # get the date
            date = datetime.datetime.now().strftime(format)
            # add the date to the format cache for unformatting purposes (don't actually reuse the date)
            self.format_cache["{date(format=" + query_ + ")}"] = date
            # replace the query with the formatted date
            text = text.replace("{date(format=" + query_ + ")}", date)

        # set the text in the text box
        self.message_edit.setText(text)
        return text

    def format_web_search_prompt(self, web_search, query='', citations=False, return_web_results=False):
        # format the web search prompt
        web_results =''
        for i, result in enumerate(web_search):
            web_results += f'[{i+1}] "{result["body"]}"\nURL: {result["href"]}\n\n'
        if return_web_results:
            return web_results
        current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if citations:
            return self.pm.replace_prompt_variables(self.web_search_and_summarize_text, {'web_results': web_results, 'query': query, 'current_date': current_date})
        else:
            return self.pm.replace_prompt_variables(self.web_search_and_summarize_no_cite_text, {'web_results': web_results, 'query': query, 'current_date': current_date})

    def updateCharCount(self):
        # Update the character count label
        text = self.message_edit.text()
        if self.prepend_first_prompt and self.parent_id is None:
            text = self.pm.replace_prompt_variables(self.prompt['prompt'], {'prompt': text})
        elif self.prepend_prompt:
            text = self.pm.replace_prompt_variables(self.prompt['prompt'], {'prompt': text})
        token_count = self.tokenizer(text, return_length=True)['length'][0]
        self.char_count_label.setText("{}/4000".format(token_count))

    def edit_continue_prompt(self):
        # make a text input dialog prepopulated with the current continue text
        text, ok = QInputDialog.getText(self, "Edit Continue Prompt", "Enter new continue prompt:", text=self.continue_text)
        if ok:
            self.continue_text = text

    def edit_web_search_and_summarize_prompt(self):
        # make a text input dialog prepopulated with the current web_search_and_summarize text
        text, ok = QInputDialog.getText(self, "Edit Web Search and Summarize Prompt", "Enter new web_search_and_summarize prompt:", text=self.web_search_and_summarize_text)
        if ok:
            self.web_search_and_summarize_text = text

    def edit_web_search_and_summarize_prompt_no_cite(self):
        # make a text input dialog prepopulated with the current web_search_and_summarize text
        text, ok = QInputDialog.getText(self, "Edit Web Search and Summarize Prompt (no citations)", "Enter new web_search_and_summarize_no_cite prompt:", text=self.web_search_and_summarize_no_cite_text)
        if ok:
            self.web_search_and_summarize_no_cite_text = text

    def promptManager(self):
        dialog = ManagePromptsDialog(pm=self.pm)
        result = dialog.exec_()

        if result == QDialog.Accepted and dialog.prompt_id is not None:
            self.prompt = self.pm.get_prompt(prompt_id=dialog.prompt_id)
            self.prepend_first_prompt = dialog.prepend_first_message
            self.prepend_prompt = dialog.prepend

            var_dict = {}
            for var in self.prompt['variables']:
                if var.lower() == 'prompt':
                    continue
                text, ok = QInputDialog.getText(self, "Prompt Variable", "Enter value for variable {}:".format(var))
                if ok:
                    var_dict[var] = text
            self.prompt['prompt'] = self.pm.replace_prompt_variables(self.prompt['prompt'], var_dict)

            self.updateCharCount(self.message_edit.text())

    def conversationManager(self):
        dialog = ManageConversationsDialog(cm=self.cm)
        result = dialog.exec_()

        if result == QDialog.Accepted:
            conversation_id = dialog.conversation_id
            self.reset_conversation_list(conversation_id)

    def reset_conversation_list(self, conversation_id):
        self.conversation_id = conversation_id
        self.parent_id = None
        self.conversations_list.clear()

        if conversation_id is not None:
            print(conversation_id)
            conversation = self.cm.get_conversation(conversation_id=conversation_id)
            for message in conversation:
                if message['message'] != None:
                    id = message['id']
                    parent_id = message['parent_id']
                    is_user = message['role'] == 'user'
                    message = message['message']
                    if is_user:
                        content = f"You: {message}"
                    else:
                        content = f"Assistant: {message}"
                    self.add_message(content, is_user=is_user, conversation_id=conversation_id, parent_id=parent_id)
                    self.parent_id = id

    def handle_text2speech_checkbox_state_change(self, state):
        if state == Qt.Checked:
            self.text2speech_enabled = True
        else:
            self.text2speech_enabled = False
        
    def add_message(self, message, is_user=True, conversation_id=None, parent_id=None):
        item = QListWidgetItem()
        widget = MessageItem(message, is_user=is_user, conversation_id=conversation_id, parent_id=parent_id)
        item.setSizeHint(widget.sizeHint())
        self.conversations_list.addItem(item)
        self.conversations_list.setItemWidget(item, widget)

    def audio2text(self, audio=None, audio_path=None):
        assert audio is not None or audio_path is not None, "Either audio or audio_path must be provided"
        if audio is not None:
            return self.audio2text_pipe(audio.squeeze())["text"]
        # get wav to text
        sample, sr = torchaudio.load(audio_path)
        # resample to 16kHz
        sample = torchaudio.transforms.Resample(sr, 16000)(sample)
        # convert to single channel
        sample = sample.mean(0, keepdim=True).squeeze(0)
        # convert to numpy
        sample = sample.numpy()

        return self.audio2text_pipe(sample)["text"]
    
    def record_audio(self, seconds=10):
        fs = 16000 # Sample rate
        seconds = seconds  # Duration of recording

        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()  # Wait until recording is finished
        return myrecording

    def play_audio(self, conversation_id, parent_id):
        music = pyglet.media.load(f"generated_audio/speech_{conversation_id}_{parent_id}.wav", streaming=False)
        music.play()

    def recordMessage(self):
        # Record audio
        recording = self.record_audio()

        # Convert the audio to text
        self.message_edit.setText(self.audio2text(audio=recording).strip())

    def continueMessage(self):
        self.message_edit.clear()
        self.sendMessage()

    def sendMessage(self):
        if self.message_edit.text() == "":
            message = self.continue_text
        else:
            # Get the message from the message edit widget
            message = self.message_edit.text()
            web_search_and_summarize_queries = re.findall(r'{web_search_and_summarize\(query=(.*?)\)}', message)
            pdf_summarize_queries = re.findall(r'{summarize_pdf\(url=(.*?)\)}', message)
            if len(web_search_and_summarize_queries) == 1 and message == f"{{web_search_and_summarize(query={web_search_and_summarize_queries[0]})}}":
                # get parameters from the query
                parameters = web_search_and_summarize_queries[0].split(',')
                # get the query
                query = parameters[0].split('=')
                if len(query) > 1:
                    query = query[1].strip()
                else:
                    query = query[0].strip()
                # get the params (may not be present)
                region = 'us-en'
                safe_search = 'Off'
                time = 'y'
                max_results = 20
                page = 1
                output = None
                download = False
                citations = False
                for parameter in parameters:
                    parameter = parameter.strip()
                    if parameter.startswith('region'):
                        region = parameter.split('=')[1].strip()
                    elif parameter.startswith('safesearch'):
                        safe_search = parameter.split('=')[1].strip()
                    elif parameter.startswith('time'):
                        time = parameter.split('=')[1].strip()
                    elif parameter.startswith('max_results'):
                        max_results = int(parameter.split('=')[1].strip())
                    elif parameter.startswith('page'):
                        page = int(parameter.split('=')[1].strip())
                    elif parameter.startswith('output'):
                        output = parameter.split('=')[1].strip()
                    elif parameter.startswith('download'):
                        download = parameter.split('=')[1].strip()
                        if download.lower() == 'true':
                            download = True
                        else:
                            download = False
                    elif parameter.startswith('citations'):
                        citations = parameter.split('=')[1].strip()
                        if citations.lower() == 'true':
                            citations = True
                        else:
                            citations = False

                # get the web search results
                web_search = self.web.search(keywords=query, region=region, safesearch=safe_search, time=time, max_results=max_results, page=page, output=output, download=download)
                message = self.format_web_search_prompt(web_search, query=query, citations=citations)
            elif len(pdf_summarize_queries) == 1 and message == f"{{summarize_pdf(url={pdf_summarize_queries[0]})}}":
                # get parameters from the query
                parameters = pdf_summarize_queries[0].split(',')
                # get the url
                url = parameters[0].split('=')
                if len(url) > 1:
                    url = url[1].strip()
                else:
                    url = url[0].strip()
                # get the params (may not be present)
                target_words = 1000
                target_words_per_page = 500
                for parameter in parameters:
                    parameter = parameter.strip()
                    if parameter.startswith('target_words') and not parameter.startswith('target_words_'):
                        target_words = int(parameter.split('=')[1].strip())
                    elif parameter.startswith('target_words_per_page'):
                        target_words_per_page = int(parameter.split('=')[1].strip())
                # get the render the summary
                message = self.summarize_pdf(url=url, target_words=target_words, target_words_per_page=target_words_per_page, render=True)
            else:
                message = self.format_text(message)

        if self.prepend_first_prompt and self.parent_id is None:
            message = self.pm.replace_prompt_variables(self.prompt['prompt'], {'prompt': message})
        elif self.prepend_prompt:
            message = self.pm.replace_prompt_variables(self.prompt['prompt'], {'prompt': message})
        
        message = message.strip()
        self.cm.add_message(message, 'user', mode=self.mode)

        # Add the message to the conversations browser widget as markdown
        # self.conversations_browser.append("You: " + message)
        self.add_message("You: " + message, is_user=True)
        
        # Clear the message edit widget
        self.message_edit.clear()

        # Show a loading message in the conversation list
        self.add_message("Assistant: Loading...", is_user=False)
        QApplication.processEvents()
        
        # Send the message to the assistant and get the response
        if self.mode == 'openai':
            chat_response = self.assistant.get_chat_response(prompt=message).choices[0].message.content
        else:
            chat_response = self.assistant.get_chat_response(prompt=message)['content']
        # Remove the previous message
        self.conversations_list.takeItem(self.conversations_list.count()-1)
        self.add_message("Assistant: " + chat_response, is_user=False)
        QApplication.processEvents()
        
        self.cm.add_message(chat_response, 'assistant')


        # Get the conversation ID and parent message ID
        self.parent_id = self.cm.parent_id
        self.conversation_id = self.cm.conversation_id
        
        if self.text2speech_enabled:
            # Generate the audio for the response
            # remove code blocks
            response = re.sub(r"\`\`\`[^`]+\`\`\`", '', chat_response)
            # remove citations
            response = re.sub(r"\[\[[0-9]+\][^\]]+\]", '', response)
            response = re.sub(r"\[[0-9]+\]\([^)\[\]]+\)", '', response)
            response = re.sub(r"\[[0-9]+\]", '', response)
            response = self.text2audio.text_to_speech(response.strip())

            # Get the audio data from the response in byte format
            audio_data = response.content

            # Create a memory stream from the audio data
            audio_stream = io.BytesIO(audio_data)

            # Load the audio data from the stream using pydub
            audio = AudioSegment.from_file(audio_stream, format="mp3")

            # Convert the audio data to WAV format
            audio.export(f"generated_audio/speech_{self.conversation_id}_{self.parent_id}.wav", format="wav")

        # Remove the previous message
        self.conversations_list.takeItem(self.conversations_list.count()-1)

        # Add the response to the conversations browser widget as markdown
        self.add_message("Assistant: " + chat_response, is_user=False, conversation_id=self.conversation_id, parent_id=self.parent_id)

        if self.text2speech_enabled:
            # Play the audio
            self.play_audio(self.conversation_id, self.parent_id)

if __name__ == '__main__':
    try:
        config = json.load(open('config.json'))
        if config['mode'] == 'openai':
            assert config['chat_config']['api_key'] != '' or config['chat_config']['api_key'] != None, 'Please provide a valid access token in the config.json file'
        elif config['mode'] == 'local':
            config['chat_config'] = config['local_chat_config']
        else:
            raise Exception('Please provide a valid mode in the config.json file (one of `openai` or `local`)')
        app = QApplication(sys.argv)
        floating_icon = FloatingIcon(chat_config=config['chat_config'], text2audio_api_key=config['text2audio_api_key'], text2audio_voice=config['text2audio_voice'], wolfram_app_id=config['wolfram_app_id'], mode=config['mode'])
        floating_icon.show()
        sys.exit(app.exec_())
    except:
        traceback.print_exc()
