import os
import json
import sqlite3
import hashlib

from .utils import sluggify


# Create a function to generate the hash of the prompt
def generate_hash(prompt):
    # Use the sha256 hash function from the hashlib library to generate the hash
    m = hashlib.sha256()
    m.update(prompt.encode('utf-8'))
    return m.hexdigest()


class PromptManager():
    """
    Class for managing prompts
    """
    def __init__(self, database: str = None) -> None:
        """
        Initialize prompt manager

        Parameters:
            database (str): SQLite database file. If None, will use the environment
                variable SQLITE_DATABASE.
        """
        if database is None:
            database = os.getenv('SQLITE_DATABASE')
            if database is None or database == '':
                database = 'prompts.db'

        self.conn = sqlite3.connect(database)
        self.conn.row_factory = sqlite3.Row
        self._initialize_tables()

    def _initialize_tables(self) -> None:
        """
        Initialize tables
        """
        cursor = self.conn.cursor()

        # Check if the chat_prompts table exists
        cursor.execute("""
            SELECT name FROM sqlite_master WHERE type='table' AND name='chat_prompts';
        """)
        chat_prompts_exists = cursor.fetchone() is not None

        # Check if the chat_prompts_tags table exists
        cursor.execute("""
            SELECT name FROM sqlite_master WHERE type='table' AND name='chat_prompts_tags';
        """)
        chat_prompts_tags_exists = cursor.fetchone() is not None

        # Check if the chat_prompts_tags_map table exists
        cursor.execute("""
            SELECT name FROM sqlite_master WHERE type='table' AND name='chat_prompts_tags_map';
        """)
        chat_prompts_tags_map_exists = cursor.fetchone() is not None

        # If any of the tables don't exist, create them
        if not chat_prompts_exists:
            cursor.execute("""
                CREATE TABLE chat_prompts (
                    id INTEGER PRIMARY KEY,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    prompt TEXT NOT NULL,
                    variables TEXT,
                    hash TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    slug TEXT NOT NULL UNIQUE,
                    notes TEXT
                );
            """)
        if not chat_prompts_tags_exists:
            cursor.execute("""
                CREATE TABLE chat_prompts_tags (
                    id INTEGER PRIMARY KEY,
                    tag TEXT NOT NULL UNIQUE
                );
            """)
        if not chat_prompts_tags_map_exists:
            cursor.execute("""
                CREATE TABLE chat_prompts_tags_map (
                    prompt_id INTEGER,
                    tag_id INTEGER,
                    PRIMARY KEY (prompt_id, tag_id),
                    FOREIGN KEY (tag_id) REFERENCES chat_prompts_tags(id),
                    FOREIGN KEY (prompt_id) REFERENCES chat_prompts(id)
                );
            """)

        # Commit the changes and close the connection
        self.conn.commit()

    def _validate_prompt(self, prompt: str, variables: list = []) -> bool:
        """
        Validate prompt

        Parameters:
            prompt (str): prompt
            variables (list): variables

        Returns:
            bool: True if valid, False if not
        """
        for variable in variables:
            if variable.upper().replace(' ', '_') + '*' not in prompt:
                return False
        return True

    def add_prompt(self, title: str, prompt: str, tags: list = None, variables: list = [], notes: str = None) -> int:
        """
        Add prompt

        Parameters:
            title (str): title
            prompt (str): prompt
            tags (list): tags
            variables (list): variables
            notes (str): notes

        Returns:
            int: prompt id
        """
        assert self._validate_prompt(prompt, variables), 'Invalid prompt'
        c = self.conn.cursor()
        try:
            c.execute('INSERT INTO chat_prompts (title, slug, prompt, hash, variables, notes) VALUES (?, ?, ?, ?, ?, ?)', (title, sluggify(title), prompt, generate_hash(prompt), None if (variables is None or variables == []) else json.dumps(variables), notes))
            prompt_id = c.lastrowid
            if tags is not None and tags != []:
                for tag in tags:
                    try:
                        c.execute('INSERT INTO chat_prompts_tags (tag) VALUES (?)', (tag,))
                        tag_id = c.lastrowid
                    except sqlite3.IntegrityError:
                        c.execute('SELECT id FROM chat_prompts_tags WHERE tag = ?', (tag,))
                        tag_id = c.fetchone()[0]
                    c.execute('INSERT INTO chat_prompts_tags_map (prompt_id, tag_id) VALUES (?, ?)', (prompt_id, tag_id))
            self.conn.commit()
            return prompt_id
        except sqlite3.IntegrityError:
            c.execute('SELECT id FROM chat_prompts WHERE slug = ?', (sluggify(title),))
            prompt_id = c.fetchone()[0]
            return prompt_id
        
    def delete_prompt(self, prompt_id):
        """
        Delete prompt

        Parameters:
            prompt_id (int): prompt id
        """
        c = self.conn.cursor()
        # delete mappings
        query = 'DELETE FROM chat_prompts_tags_map WHERE prompt_id = ?'
        c.execute(query, (prompt_id,))
        # delete prompt
        query = 'DELETE FROM chat_prompts WHERE id = ?'
        c.execute(query, (prompt_id,))
        self.conn.commit()

    def update_prompt(self, prompt_id, title, prompt, tags=None, variables=[], notes=None):
        """
        Update prompt

        Parameters:
            prompt_id (int): prompt id
            title (str): title
            prompt (str): prompt
            tags (list): tags
            variables (list): variables
            notes (str): notes
        """
        c = self.conn.cursor()
        # update prompt
        query = 'UPDATE chat_prompts SET title = ?, slug = ?, prompt = ?, hash = ?, variables = ?, notes = ? WHERE id = ?'
        c.execute(query, (title, sluggify(title), prompt, generate_hash(prompt), None if (variables == []) else json.dumps(variables), notes, prompt_id))
        # delete mappings
        query = 'DELETE FROM chat_prompts_tags_map WHERE prompt_id = ?'
        c.execute(query, (prompt_id,))
        # add mappings
        if tags is not None and tags != []:
            for tag in tags:
                try:
                    query = 'INSERT INTO chat_prompts_tags (tag) VALUES (?)'
                    c.execute(query, (tag,))
                    tag_id = c.lastrowid
                except sqlite3.IntegrityError:
                    query = 'SELECT id FROM chat_prompts_tags WHERE tag = ?'
                    c.execute(query, (tag,))
                    tag_id = c.fetchone()[0]
                try:
                    query = 'INSERT INTO chat_prompts_tags_map (prompt_id, tag_id) VALUES (?, ?)'
                    c.execute(query, (prompt_id, tag_id))
                except:
                    continue
        self.conn.commit()
    
    def get_prompt(self, prompt_id: int = None, prompt_slug: str = None) -> dict:
        """
        Get prompt

        Parameters:
            prompt_id (int): prompt id
            prompt_slug (str): prompt slug

        Returns:
            dict: prompt
        """
        assert prompt_id is not None or prompt_slug is not None, 'prompt_id and prompt_slug are None'
        c = self.conn.cursor()
        if prompt_id is not None:
            query = '''
            SELECT 
                cp.id, 
                cp.title, 
                cp.slug, 
                cp.prompt, 
                cp.variables, 
                cp.notes, 
                group_concat(cpt.tag) AS tags 
            FROM chat_prompts cp 
            INNER JOIN chat_prompts_tags_map cptm ON cp.id = cptm.prompt_id 
            INNER JOIN chat_prompts_tags cpt ON cptm.tag_id = cpt.id 
            WHERE cp.id = ?
            GROUP BY cp.id
            '''
            prompt = c.execute(query, (prompt_id,)).fetchone()
        elif prompt_slug is not None:
            query = '''
            SELECT 
                cp.id, 
                cp.title, 
                cp.slug, 
                cp.prompt, 
                cp.variables, 
                cp.notes, 
                group_concat(cpt.tag) AS tags 
            FROM chat_prompts cp 
            INNER JOIN chat_prompts_tags_map cptm ON cp.id = cptm.prompt_id 
            INNER JOIN chat_prompts_tags cpt ON cptm.tag_id = cpt.id 
            WHERE cp.slug = ?
            GROUP BY cp.id
            '''
            prompt = c.execute(query, (prompt_slug,)).fetchone()
        if prompt is not None:
            # convert to dict
            prompt = dict(zip([column[0] for column in c.description], prompt))
            prompt['variables'] = json.loads(prompt['variables'])
            prompt['tags'] = prompt['tags'].split(',')
        return prompt


    def replace_prompt_variables(self, prompt: str, variables: dict) -> str:
        """
        Replace prompt variables

        Parameters:
            prompt (str): prompt
            variables (dict): variables

        Returns:
            str: prompt with variables replaced
        """
        assert self._validate_prompt(prompt, variables.keys()), 'Invalid prompt'
        for variable, value in variables.items():
            prompt = prompt.replace(variable.upper().replace(' ', '_') + '*', str(value))
        return prompt
    
    def get_prompts(self, tags: list = None, tag_filter_mode: str = 'any', offset: int = 0, limit: int = 20) -> list:
        """
        Get prompts

        Parameters:
            tags (list): tags
            tag_filter_mode (str): tag filter mode

        Returns:
            list: prompts
        """
        c = self.conn.cursor()
        if tags is None or tags == [] or tags[0] == '':
            query = '''
            SELECT cp.id, cp.title, cp.slug, cp.prompt, cp.variables, group_concat(cpt.tag) AS tags, cp.notes
            FROM chat_prompts cp 
            INNER JOIN chat_prompts_tags_map cptm ON cp.id = cptm.prompt_id 
            INNER JOIN chat_prompts_tags cpt ON cptm.tag_id = cpt.id
            GROUP BY cp.id
            ORDER BY cp.title
            LIMIT ?, ?
            '''
            prompts = c.execute(query, (offset, limit))
        else:
            tags = [tag.lower() for tag in tags]
            if tag_filter_mode == 'any':
                query = '''
                SELECT cp.id, cp.title, cp.slug, cp.prompt, cp.variables, cp.notes, 
                    group_concat(cpt.tag, ',') AS tags,
                    (SELECT group_concat(cpt2.tag, ',') 
                        FROM chat_prompts_tags cpt2 
                        INNER JOIN chat_prompts_tags_map cptm2 ON cpt2.id = cptm2.tag_id 
                        WHERE cpt2.tag in ({}) AND cptm2.prompt_id = cp.id) AS matched_tags
                FROM chat_prompts cp 
                INNER JOIN chat_prompts_tags_map cptm ON cp.id = cptm.prompt_id 
                INNER JOIN chat_prompts_tags cpt ON cptm.tag_id = cpt.id
                GROUP BY cp.id
                HAVING length(matched_tags) - length(replace(matched_tags, ',', '')) + 1 >= {}
                ORDER BY cp.title
                LIMIT ?, ?
                '''.format(', '.join('?' for tag in tags), 1)
                prompts = c.execute(query, (*tags, offset, limit))
            elif tag_filter_mode == 'all':
                query = '''
                SELECT cp.id, cp.title, cp.slug, cp.prompt, cp.variables, cp.notes, 
                    group_concat(cpt.tag, ',') AS tags,
                    (SELECT group_concat(cpt2.tag, ',') 
                        FROM chat_prompts_tags cpt2 
                        INNER JOIN chat_prompts_tags_map cptm2 ON cpt2.id = cptm2.tag_id 
                        WHERE cpt2.tag in ({}) AND cptm2.prompt_id = cp.id) AS matched_tags
                FROM chat_prompts cp 
                INNER JOIN chat_prompts_tags_map cptm ON cp.id = cptm.prompt_id 
                INNER JOIN chat_prompts_tags cpt ON cptm.tag_id = cpt.id
                GROUP BY cp.id
                HAVING length(matched_tags) - length(replace(matched_tags, ',', '')) + 1 = {}
                ORDER BY cp.title
                LIMIT ?, ?
                '''.format(', '.join('?' for tag in tags), len(tags))
                prompts = c.execute(query, (*tags, offset, limit))
        prompts = prompts.fetchall()
        # convert prompts to list of dicts
        prompts = [dict(zip([column[0] for column in c.description], prompt)) for prompt in prompts]
        for prompt in prompts:
            prompt['variables'] = json.loads(prompt['variables'])
            if prompt.get('matched_tags') is not None:
                prompt['matched_tags'] = prompt['matched_tags'].split(',')
            prompt['tags'] = prompt['tags'].split(',')

        return prompts
    
    def get_prompts_count(self, tags: list = None, tag_filter_mode: str = 'any') -> int:
        """
        Get prompts count

        Parameters:
            tags (list): tags
            tag_filter_mode (str): tag filter mode

        Returns:
            int: prompts count
        """
        c = self.conn.cursor()
        if tags is None or tags == [] or tags[0] == '':
            query = '''
            SELECT COUNT(*) FROM chat_prompts
            '''
            count = c.execute(query).fetchone()[0]
        else:
            tags = [tag.lower() for tag in tags]
            if tag_filter_mode == 'any':
                query = '''
                SELECT COUNT(DISTINCT cp.id)
                FROM chat_prompts cp 
                INNER JOIN chat_prompts_tags_map cptm ON cp.id = cptm.prompt_id 
                INNER JOIN chat_prompts_tags cpt ON cptm.tag_id = cpt.id
                WHERE cpt.tag in ({})
                '''.format(', '.join('?' for tag in tags))
                count = c.execute(query, (*tags,)).fetchone()
            elif tag_filter_mode == 'all':
                query = '''
                SELECT COUNT(DISTINCT cp.id)
                FROM chat_prompts cp 
                INNER JOIN chat_prompts_tags_map cptm ON cp.id = cptm.prompt_id 
                INNER JOIN chat_prompts_tags cpt ON cptm.tag_id = cpt.id
                WHERE cpt.tag in ({})
                GROUP BY cp.id
                HAVING count(cpt.tag) = {}
                '''.format(', '.join('?' for tag in tags), len(tags))
                count = c.execute(query, (*tags,)).fetchone()
        if count is None:
            count = 0
        if not isinstance(count, int):
            count = count[0]
        return count
    
    def get_tags(self) -> list:
        """
        Get tags

        Returns:
            list: tags
        """
        c = self.conn.cursor()
        query = '''
        SELECT tag FROM chat_prompts_tags
        ORDER BY tag
        '''
        tags = c.execute(query).fetchall()
        tags = [tag[0] for tag in tags]
        return tags
