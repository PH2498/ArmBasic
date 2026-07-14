"""
Flask API 路由
实现用户认证相关的 RESTful API 接口
"""
from flask import Flask, request, jsonify
from functools import wraps
from .auth_service import get_auth_service
from .utils.jwt_handler import JWTHandler


# Flask 应用实例
app = Flask(__name__)


def token_required(f):
    """Token 验证装饰器
    
    验证请求头中的 Authorization Token
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # 从请求头获取 Token
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
        
        if not token:
            return jsonify({
                'code': 401,
                'message': '缺少认证令牌'
            }), 401
        
        # 验证 Token
        auth_service = get_auth_service()
        valid, user_data = auth_service.verify_token(token)
        
        if not valid:
            return jsonify({
                'code': 401,
                'message': '令牌无效或已过期'
            }), 401
        
        # 将用户信息添加到请求上下文
        request.current_user = user_data
        
        return f(*args, **kwargs)
    
    return decorated


@app.route('/api/auth/register', methods=['POST'])
def register():
    """用户注册接口"""
    data = request.get_json()
    
    if not data:
        return jsonify({
            'code': 400,
            'message': '请求体不能为空'
        }), 400
    
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({
            'code': 400,
            'message': '用户名和密码不能为空'
        }), 400
    
    auth_service = get_auth_service()
    success, message, user_data = auth_service.register_user(username, password)
    
    if success:
        return jsonify({
            'code': 200,
            'message': message,
            'data': user_data
        }), 200
    else:
        return jsonify({
            'code': 400,
            'message': message
        }), 400


@app.route('/api/auth/login', methods=['POST'])
def login():
    """用户登录接口"""
    data = request.get_json()
    
    if not data:
        return jsonify({
            'code': 400,
            'message': '请求体不能为空'
        }), 400
    
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({
            'code': 400,
            'message': '用户名和密码不能为空'
        }), 400
    
    auth_service = get_auth_service()
    success, message, auth_data = auth_service.login(username, password)
    
    if success:
        return jsonify({
            'code': 200,
            'message': message,
            'data': auth_data
        }), 200
    else:
        return jsonify({
            'code': 401,
            'message': message
        }), 401


@app.route('/api/auth/logout', methods=['POST'])
@token_required
def logout():
    """用户登出接口"""
    token = request.headers.get('Authorization').split(' ')[1]
    
    auth_service = get_auth_service()
    success, message = auth_service.logout(token)
    
    return jsonify({
        'code': 200 if success else 400,
        'message': message
    }), 200 if success else 400


@app.route('/api/auth/refresh', methods=['POST'])
def refresh():
    """刷新令牌接口"""
    data = request.get_json()
    
    if not data:
        return jsonify({
            'code': 400,
            'message': '请求体不能为空'
        }), 400
    
    refresh_token = data.get('refresh_token')
    
    if not refresh_token:
        return jsonify({
            'code': 400,
            'message': '刷新令牌不能为空'
        }), 400
    
    auth_service = get_auth_service()
    success, message, token_data = auth_service.refresh_token(refresh_token)
    
    if success:
        return jsonify({
            'code': 200,
            'message': message,
            'data': token_data
        }), 200
    else:
        return jsonify({
            'code': 401,
            'message': message
        }), 401


@app.route('/api/auth/profile', methods=['GET'])
@token_required
def get_profile():
    """获取用户信息接口"""
    user_id = request.current_user.get('user_id')
    
    auth_service = get_auth_service()
    success, message, user_data = auth_service.get_user_profile(user_id)
    
    if success:
        return jsonify({
            'code': 200,
            'message': message,
            'data': user_data
        }), 200
    else:
        return jsonify({
            'code': 404,
            'message': message
        }), 404


@app.route('/api/auth/profile', methods=['PUT'])
@token_required
def update_profile():
    """更新用户信息接口"""
    # 基础实现，实际应用需要扩展
    return jsonify({
        'code': 200,
        'message': '功能开发中'
    }), 200


@app.route('/api/auth/face-login', methods=['POST'])
def face_login():
    """人脸识别登录接口"""
    # 基础实现，需要集成 FaceRecognitionModule
    return jsonify({
        'code': 501,
        'message': '人脸识别登录功能开发中'
    }), 501


@app.route('/api/auth/face/register', methods=['POST'])
@token_required
def register_face():
    """注册人脸接口"""
    # 基础实现，需要集成 FaceRecognitionModule
    return jsonify({
        'code': 501,
        'message': '人脸注册功能开发中'
    }), 501


@app.errorhandler(404)
def not_found(error):
    """404 错误处理"""
    return jsonify({
        'code': 404,
        'message': '资源不存在'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """500 错误处理"""
    return jsonify({
        'code': 500,
        'message': '服务器内部错误'
    }), 500


def create_app(config_class=None):
    """创建 Flask 应用实例
    
    Args:
        config_class: 配置类
        
    Returns:
        Flask 应用实例
    """
    app_instance = Flask(__name__)
    
    if config_class:
        app_instance.config.from_object(config_class)
    
    # 注册路由
    from .routes import app as routes_app
    app_instance.register_blueprint(routes_app)
    
    return app_instance


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)