import os
import sqlite3
import uuid


class ConversationManager():
    """
    Conversation manager
    """
    def __init__(self, database: str = None) -> None:
        """
        Initialize conversation manager

        Parameters:
            database (str): SQLite database file. If None, will use the environment
                variable SQLITE_DATABASE.
        """
        if database is None:
            database = os.getenv('SQLITE_DATABASE')
            if database is None or database == '':
                database = 'conversations.db'

        self.conn = sqlite3.connect(database)
        self.conn.row_factory = sqlite3.Row
        self._initialize_tables()
        self.conversation_id = None
        self.parent_id = None

    def _initialize_tables(self) -> None:
        """
        Initialize tables
        """
        c = self.conn.cursor()
        query = '''
        CREATE TABLE IF NOT EXISTS messages (
            id UUID PRIMARY KEY,
            parent_id UUID,
            conversation_id UUID,
            message TEXT,
            role TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        '''
        c.execute(query)
        query = '''
        CREATE TABLE IF NOT EXISTS conversations (
            id UUID PRIMARY KEY,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        '''
        c.execute(query)
        self.conn.commit()

    def set_conversation_id(self, conversation_id: uuid.UUID) -> None:
        """
        Set conversation ID

        Parameters:
            conversation_id (uuid.UUID): conversation ID
        """
        self.conversation_id = conversation_id

    def add_conversation(self, title: str = None) -> uuid.UUID:
        """
        Add conversation

        Parameters:
            title (str): title

        Returns:
            uuid.UUID: conversation ID
        """
        c = self.conn.cursor()
        conversation_id = uuid.uuid4()
        query = '''
        INSERT INTO conversations (id, title) VALUES (?, ?)
        '''
        if title == None:
            title = 'New Chat'
        c.execute(query, (str(conversation_id), title))
        self.conn.commit()
        self.conversation_id = conversation_id
        return conversation_id
    
    def add_message(self, message: str, role: str, parent_id: uuid.UUID = None, conversation_id: uuid.UUID = None, mode: str = 'openai') -> uuid.UUID:
        """
        Add message

        Parameters:
            message (str): message
            role (str): role (user/human or assistant)
            parent_id (uuid.UUID): parent ID
            conversation_id (uuid.UUID): conversation ID

        Returns:
            uuid.UUID: message ID
        """
        assert role in ['user', 'human', 'assistant'], "Role must be 'user' or 'assistant'"
        if mode == 'local' and role == 'user':
            role = 'human'
        c = self.conn.cursor()
        message_id = uuid.uuid4()
        if parent_id == None:
            parent_id = self.parent_id
        if conversation_id == None:
            conversation_id = self.conversation_id
            if conversation_id == None:
                conversation_id = self.add_conversation()
        query = '''
        INSERT INTO messages (id, parent_id, conversation_id, message, role) VALUES (?, ?, ?, ?, ?)
        '''
        c.execute(query, (str(message_id), str(parent_id), str(conversation_id), message, role))
        self.conn.commit()
        self.parent_id = message_id
        return message_id
    
    def rename_conversation(self, conversation_id: uuid.UUID, title: str) -> None:
        """
        Rename conversation

        Parameters:
            conversation_id (uuid.UUID): conversation ID
            title (str): title
        """
        c = self.conn.cursor()
        query = '''
        UPDATE conversations SET title = ? WHERE id = ?
        '''
        c.execute(query, (title, str(conversation_id)))
        self.conn.commit()
    
    def get_conversation(self, conversation_id: uuid.UUID) -> dict:
        """
        Get conversation

        Parameters:
            conversation_id (uuid.UUID): conversation ID

        Returns:
            dict: conversation
        """
        c = self.conn.cursor()
        query = '''
        SELECT * FROM messages WHERE conversation_id = ?
        '''
        c.execute(query, (str(conversation_id),))
        conversation = c.fetchall()
        # create a dictionary of messages, with column names as keys
        conversation = [{column: message[i] for i, column in enumerate(message.keys())} for message in conversation]
        # create a dictionary of messages, with message parent_ids as keys
        messages = {message['parent_id']: message for message in conversation}

        # create a list of ordered messages
        ordered_conversation = []
        try:
            root_message = messages[None]
        except:
            root_message = messages['None']
        ordered_conversation.append(root_message)
        next_id = root_message['id']
        while next_id in messages:
            # find the next message in the chain by looking for the message with a parent_id equal to the current message's id
            next_message = messages[next_id]
            ordered_conversation.append(next_message)
            next_id = next_message['id']

        self.parent_id = next_id

        return ordered_conversation
    
    def get_messages(self, conversation_id: uuid.UUID) -> dict:
        """
        Get conversation

        Parameters:
            conversation_id (uuid.UUID): conversation ID

        Returns:
            dict: conversation
        """
        return self.get_conversation(conversation_id)
    
    def get_conversations(self) -> list:
        """
        Get conversations

        Returns:
            list: conversations
        """
        c = self.conn.cursor()
        query = '''
        SELECT * FROM conversations
        ORDER BY created_at DESC
        '''
        c.execute(query)
        conversations = c.fetchall()
        # create a dictionary of messages, with column names as keys
        conversations = [{column: conversation[i] for i, column in enumerate(conversation.keys())} for conversation in conversations]
        return conversations
    
    def delete_conversation(self, conversation_id: uuid.UUID) -> None:
        """
        Delete conversation

        Parameters:
            conversation_id (uuid.UUID): conversation ID
        """
        c = self.conn.cursor()
        query = '''
        DELETE FROM conversations WHERE id = ?
        '''
        c.execute(query, (str(conversation_id),))
        self.conn.commit()

    def delete_message(self, message_id: uuid.UUID) -> None:
        """
        Delete message

        Parameters:
            message_id (uuid.UUID): message ID
        """
        c = self.conn.cursor()
        query = '''
        DELETE FROM messages WHERE id = ?
        '''
        c.execute(query, (str(message_id),))
        self.conn.commit()

    def delete_messages(self, conversation_id: uuid.UUID) -> None:
        """
        Delete messages

        Parameters:
            conversation_id (uuid.UUID): conversation ID
        """
        c = self.conn.cursor()
        query = '''
        DELETE FROM messages WHERE conversation_id = ?
        '''
        c.execute(query, (str(conversation_id),))
        self.conn.commit()

    def delete_all_messages(self) -> None:
        """
        Delete all messages
        """
        c = self.conn.cursor()
        query = '''
        DELETE FROM messages
        '''
        c.execute(query)
        self.conn.commit()

    def delete_all_conversations(self) -> None:
        """
        Delete all conversations
        """
        c = self.conn.cursor()
        # delete all messages
        self.delete_all_messages()
        query = '''
        DELETE FROM conversations
        '''
        c.execute(query)
        self.conn.commit()
