"""Flask API 路由"""
from flask import Flask, request, jsonify
from functools import wraps

from auth_service import auth_service


def create_app():
    """创建 Flask 应用"""
    app = Flask(__name__)
    
    # 注册路由
    app.route('/api/auth/register', methods=['POST'])(register)
    app.route('/api/auth/login', methods=['POST'])(login)
    app.route('/api/auth/logout', methods=['POST'])(logout)
    app.route('/api/auth/profile', methods=['GET'])(get_profile)
    
    return app


def token_required(f):
    """Token 验证装饰器"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({'code': 401, 'message': '缺少认证 Token'}), 401
        
        valid, user_id = auth_service.verify_token(token)
        if not valid:
            return jsonify({'code': 401, 'message': 'Token 无效或已过期'}), 401
        
        request.user_id = user_id
        return f(*args, **kwargs)
    return decorated


def register():
    """用户注册"""
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '')
    
    success, message, user_id = auth_service.register(username, password)
    
    if success:
        return jsonify({'code': 200, 'message': message, 'data': {'user_id': user_id}}), 200
    return jsonify({'code': 400, 'message': message}), 400


def login():
    """用户登录"""
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '')
    
    success, message, result = auth_service.login(username, password)
    
    if success:
        return jsonify({'code': 200, 'message': message, 'data': result}), 200
    return jsonify({'code': 401, 'message': message}), 401


def logout():
    """用户登出"""
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    
    success, message = auth_service.logout(token)
    
    if success:
        return jsonify({'code': 200, 'message': message}), 200
    return jsonify({'code': 400, 'message': message}), 400


@token_required
def get_profile():
    """获取用户信息"""
    from user_model import User, get_session
    
    session = get_session()
    try:
        user = session.query(User).filter_by(id=request.user_id).first()
        if not user:
            return jsonify({'code': 404, 'message': '用户不存在'}), 404
        
        return jsonify({
            'code': 200,
            'data': {
                'user_id': user.id,
                'username': user.username,
                'created_at': user.created_at.isoformat()
            }
        }), 200
    finally:
        session.close()


# 创建应用实例
app = create_app()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)