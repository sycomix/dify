import base64
import datetime
import hashlib
import hmac
import json
import queue
from typing import Optional
from urllib.parse import urlparse
import ssl
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time

import websocket


class SparkLLMClient:
    def __init__(self, model_name: str, app_id: str, api_key: str, api_secret: str, api_domain: Optional[str] = None):
        domain = 'spark-api.xf-yun.com'
        endpoint = 'chat'
        if api_domain:
            domain = api_domain
            if model_name == 'spark-v3':
                endpoint = 'multimodal'

        model_api_configs = {
            'spark': {
                'version': 'v1.1',
                'chat_domain': 'general'
            },
            'spark-v2': {
                'version': 'v2.1',
                'chat_domain': 'generalv2'
            },
            'spark-v3': {
                'version': 'v3.1',
                'chat_domain': 'generalv3'
            }
        }

        api_version = model_api_configs[model_name]['version']

        self.chat_domain = model_api_configs[model_name]['chat_domain']
        self.api_base = f"wss://{domain}/{api_version}/{endpoint}"
        self.app_id = app_id
        self.ws_url = self.create_url(
            urlparse(self.api_base).netloc,
            urlparse(self.api_base).path,
            self.api_base,
            api_key,
            api_secret
        )

        self.queue = queue.Queue()
        self.blocking_message = ''

    def create_url(self, host: str, path: str, api_base: str, api_key: str, api_secret: str) -> str:
        # generate timestamp by RFC1123
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        signature_origin = f"host: {host}" + "\n"
        signature_origin += f"date: {date}" + "\n"
        signature_origin += f"GET {path} HTTP/1.1"

        # encrypt using hmac-sha256
        signature_sha = hmac.new(api_secret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()

        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = f'api_key="{api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        v = {
            "authorization": authorization,
            "date": date,
            "host": host
        }
        return f'{api_base}?{urlencode(v)}'

    def run(self, messages: list, user_id: str,
            model_kwargs: Optional[dict] = None, streaming: bool = False):
        websocket.enableTrace(False)
        ws = websocket.WebSocketApp(
            self.ws_url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        ws.messages = messages
        ws.user_id = user_id
        ws.model_kwargs = model_kwargs
        ws.streaming = streaming
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

    def on_error(self, ws, error):
        self.queue.put({
            'status_code': error.status_code,
            'error': error.resp_body.decode('utf-8')
        })
        ws.close()

    def on_close(self, ws, close_status_code, close_reason):
        self.queue.put({'done': True})

    def on_open(self, ws):
        self.blocking_message = ''
        data = json.dumps(self.gen_params(
            messages=ws.messages,
            user_id=ws.user_id,
            model_kwargs=ws.model_kwargs
        ))
        ws.send(data)

    def on_message(self, ws, message):
        data = json.loads(message)
        code = data['header']['code']
        if code != 0:
            self.queue.put({
                'status_code': 400,
                'error': f"Code: {code}, Error: {data['header']['message']}"
            })
            ws.close()
        else:
            choices = data["payload"]["choices"]
            status = choices["status"]
            content = choices["text"][0]["content"]
            if ws.streaming:
                self.queue.put({'data': content})
            else:
                self.blocking_message += content

            if status == 2:
                if not ws.streaming:
                    self.queue.put({'data': self.blocking_message})
                ws.close()

    def gen_params(self, messages: list, user_id: str,
                   model_kwargs: Optional[dict] = None) -> dict:
        data = {
            "header": {
                "app_id": self.app_id,
                "uid": user_id
            },
            "parameter": {
                "chat": {
                    "domain": self.chat_domain
                }
            },
            "payload": {
                "message": {
                    "text": messages
                }
            }
        }

        if model_kwargs:
            data['parameter']['chat'].update(model_kwargs)

        return data

    def subscribe(self):
        while True:
            content = self.queue.get()
            if 'error' in content:
                if content['status_code'] == 401:
                    raise SparkError('[Spark] The credentials you provided are incorrect. '
                                     'Please double-check and fill them in again.')
                elif content['status_code'] == 403:
                    raise SparkError("[Spark] Sorry, the credentials you provided are access denied. "
                                     "Please try again after obtaining the necessary permissions.")
                else:
                    raise SparkError(f"[Spark] code: {content['status_code']}, error: {content['error']}")

            if 'data' not in content:
                break
            yield content


class SparkError(Exception):
    pass
