from __future__ import absolute_import, unicode_literals
import os
from flask import Flask, request, abort, render_template, send_from_directory
from wechatpy import parse_message, create_reply
from wechatpy.utils import check_signature
from wechatpy.exceptions import (
    InvalidSignatureException,
    InvalidAppIdException,
)

import wx_core as core


# set token or get from environments
TOKEN = os.getenv('WECHAT_TOKEN', 'chenr18')
APPID = os.getenv('WECHAT_APP_ID', 'wxe433134f58849e5b')
AES_KEY = os.getenv('WECHAT_AES_KEY', '3mqnheGLFrn7zWtvxtR97AqsVMog8rIIzjaswX7Aj5n')

app = Flask(__name__)


@app.route('/index')
def index():
    host = request.url_root
    return render_template('index.html', host=host)


@app.route('/', methods=['GET', 'POST'])
def wechat():
    signature = request.args.get('signature', '')
    timestamp = request.args.get('timestamp', '')
    nonce = request.args.get('nonce', '')
    encrypt_type = request.args.get('encrypt_type', 'raw')
    msg_signature = request.args.get('msg_signature', '')
    try:
        check_signature(TOKEN, signature, timestamp, nonce)
    except InvalidSignatureException:
        abort(403)
    if request.method == 'GET':
        echo_str = request.args.get('echostr', '')
        return echo_str

    # POST request
    if encrypt_type == 'raw':
        # plaintext mode
        msg = parse_message(request.data)
        reply = core.msg_handler(msg)
        return reply.render()
    else:
        # encryption mode
        from wechatpy.crypto import WeChatCrypto

        crypto = WeChatCrypto(TOKEN, AES_KEY, APPID)
        try:
            msg = crypto.decrypt_message(
                request.data,
                msg_signature,
                timestamp,
                nonce
            )
        except (InvalidSignatureException, InvalidAppIdException):
            abort(403)
        else:
            msg = parse_message(msg)
            reply = core.msg_handler(msg)
            return crypto.encrypt_message(reply.render(), nonce, timestamp)

if __name__ == '__main__':
    app.run('0.0.0.0', 8080, debug=True)
