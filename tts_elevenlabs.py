import requests

class TTSElevenlabs():
    """
    Class for text2audio conversion using elevenlabs api
    """
    def __init__(self, api_key: str) -> None:
        """
        Initiate the class

        Parameters:
            api_key (str): api key for elevenlabs api
        """
        self.headers =  {
            'accept': 'application/json',
            'xi-api-key': api_key,
        }
        self.endpoint_prefix = 'https://api.elevenlabs.io/v1'
        self.endpoints = {
            'get_subscription_info': '/user/subscription',
            'get_user_info': '/user',
            'get_voices': '/voices',
            'get_default_voice_settings': '/default',
            'get_voice_settings': '/voices/{voice_id}/settings',
            'get_voice': '/voices/{voice_id}',
            'delete_voice': '/voices/{voice_id}',
            'edit_voice_settings': '/voices/{voice_id}/settings/edit',
            'add_voice': '/voices/add',
            'edit_voice': '/voices/{voice_id}/edit',
            'text_to_speech': '/text-to-speech/{voice_id}',
            'text_to_speech_stream': '/text-to-speech/{voice_id}/stream',
            'delete sample': '/voices/{voice_id}/samples/{sample_id}',
            'get_sample': '/voices/{voice_id}/samples/{sample_id}/audio',
            'get_history': '/history',
            'get_audio_from_history': '/history/{history_id}/audio',
            'delete_history': '/history/{history_id}',
            'download_history': '/history/download',
        }

        self.active_voice = None

    def get_subscription_info(self) -> dict:
        """
        Get subscription info

        Returns:
            dict: subscription info
        """
        response = requests.get(
            self.endpoint_prefix + self.endpoints['get_subscription_info'],
            headers=self.headers
        )
        return response.json()

    def get_user_info(self) -> dict:
        """
        Get user info

        Returns:
            dict: user info
        """
        response = requests.get(
            self.endpoint_prefix + self.endpoints['get_user_info'],
            headers=self.headers
        )
        return response.json()

    def get_voices(self) -> dict:
        """
        Get voices

        Returns:
            dict: voices
        """
        response = requests.get(
            self.endpoint_prefix + self.endpoints['get_voices'],
            headers=self.headers
        )
        return response.json()

    def get_default_voice_settings(self) -> dict:
        """
        Get default voice settings

        Returns:
            dict: default voice settings
        """
        response = requests.get(
            self.endpoint_prefix + self.endpoints['get_default_voice_settings'],
            headers=self.headers
        )
        return response.json()

    def get_voice_settings(self, voice_id: str = None) -> dict:
        """
        Get voice settings

        Parameters:
            voice_id (str): voice id (if None, active voice will be used)

        Returns:
            dict: voice settings
        """
        if voice_id is None:
            voice_id = self.active_voice
        response = requests.get(
            self.endpoint_prefix + self.endpoints['get_voice_settings'].format(voice_id=voice_id),
            headers=self.headers
        )
        return response.json()

    def get_voice(self, voice_id: str = None) -> dict:
        """
        Get voice

        Parameters:
            voice_id (str): voice id (if None, active voice will be used)
        """
        if voice_id is None:
            voice_id = self.active_voice
        response = requests.get(
            self.endpoint_prefix + self.endpoints['get_voice'].format(voice_id=voice_id),
            headers=self.headers
        )
        return response.json()

    def delete_voice(self, voice_id: str) -> dict:
        """
        Delete voice

        Parameters:
            voice_id (str): voice id

        Returns:
            dict: response
        """
        response = requests.delete(
            self.endpoint_prefix + self.endpoints['delete_voice'].format(voice_id=voice_id),
            headers=self.headers
        )
        return response.json()

    def edit_voice_settings(self, voice_id: str = None, data: dict = {}) -> dict:
        """
        Edit voice settings

        Parameters:
            voice_id (str): voice id (if None, active voice will be used)
            data (dict): data to be sent

        Returns:
            dict: response
        """
        if voice_id is None:
            voice_id = self.active_voice
        response = requests.post(
            self.endpoint_prefix + self.endpoints['edit_voice_settings'].format(voice_id=voice_id),
            headers=self.headers,
            json=data
        )
        return response.json()

    def add_voice(self, data: dict) -> dict:
        """
        Add voice

        Parameters:
            data (dict): data to be sent

        Returns:
            dict: response
        """
        response = requests.post(
            self.endpoint_prefix + self.endpoints['add_voice'],
            headers=self.headers,
            json=data
        )
        return response.json()

    def edit_voice(self, voice_id: str = None, data: dict = {}) -> dict:
        """
        Edit voice
        """
        if voice_id is None:
            voice_id = self.active_voice
        response = requests.post(
            self.endpoint_prefix + self.endpoints['edit_voice'].format(voice_id=voice_id),
            headers=self.headers,
            json=data
        )
        return response.json()

    def text_to_speech(self, text: str, voice_id: str = None) -> dict:
        """
        Text to speech
        """
        if voice_id is None:
            voice_id = self.active_voice
        response = requests.post(
            self.endpoint_prefix + self.endpoints['text_to_speech'].format(voice_id=voice_id),
            headers=self.headers,
            json={'text': text}
        )
        return response

    def text_to_speech_stream(self, text: str, voice_id: str = None) -> dict:
        """
        Text to speech stream
        """
        if voice_id is None:
            voice_id = self.active_voice
        response = requests.post(
            self.endpoint_prefix + self.endpoints['text_to_speech_stream'].format(voice_id=voice_id),
            headers=self.headers,
            json={'text': text}
        )
        return response

    def delete_sample(self, sample_id: str, voice_id: str = None) -> dict:
        """
        Delete sample
        """
        if voice_id is None:
            voice_id = self.active_voice
        response = requests.delete(
            self.endpoint_prefix + self.endpoints['delete sample'].format(voice_id=voice_id, sample_id=sample_id),
            headers=self.headers
        )
        return response.json()

    def get_sample(self, sample_id: str, voice_id: str = None) -> dict:
        """
        Get sample
        """
        if voice_id is None:
            voice_id = self.active_voice
        response = requests.get(
            self.endpoint_prefix + self.endpoints['get_sample'].format(voice_id=voice_id, sample_id=sample_id),
            headers=self.headers
        )
        return response.json()

    def get_history(self) -> dict:
        """
        Get history
        """
        response = requests.get(
            self.endpoint_prefix + self.endpoints['get_history'],
            headers=self.headers
        )
        return response.json()

    def get_audio_from_history(self, history_id: str) -> dict:
        """
        Get audio from history
        """
        response = requests.get(
            self.endpoint_prefix + self.endpoints['get_audio_from_history'].format(history_id=history_id),
            headers=self.headers
        )
        return response.json()

    def delete_history(self, history_id: str) -> dict:
        """
        Delete history
        """
        response = requests.delete(
            self.endpoint_prefix + self.endpoints['delete_history'].format(history_id=history_id),
            headers=self.headers
        )
        return response.json()

    def download_history(self, data: dict) -> dict:
        """
        Download history
        """
        response = requests.post(
            self.endpoint_prefix + self.endpoints['download_history'],
            headers=self.headers,
            json=data
        )
        return response.json()

    def set_active_voice(self, voice_id: str = None, voice_name: str = None) -> None:
        """
        Set active voice using voice ID (if not provided, will fallback to using voice name)

        Parameters:
            voice_id (str): Voice ID
            voice_name (str): Voice name
        """
        if voice_id != None:
            self.active_voice = voice_id
        elif voice_name != None:
            voices = self.get_voices()
            found_voice = False
            for voice in voices['voices']:
                if voice['name'] == voice_name:
                    self.active_voice = voice['voice_id']
                    found_voice = True
                    break
            if found_voice == False:
                raise Exception('Voice name not found')
        else:
            raise Exception('No voice ID or name provided')
