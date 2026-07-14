"""
基础功能测试脚本
验证用户注册、登录、Token 验证流程
"""
import sys
import os

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from user_model import init_db
from auth_service import get_auth_service
from utils.jwt_handler import JWTHandler


def test_register_and_login():
    """测试注册和登录流程"""
    print("=" * 60)
    print("用户登录功能 - 基础验证")
    print("=" * 60)
    
    # 初始化数据库
    print("\n[1] 初始化数据库...")
    init_db()
    print("    ✓ 数据库初始化成功")
    
    # 获取认证服务
    auth_service = get_auth_service()
    
    # 测试用户注册
    print("\n[2] 测试用户注册...")
    test_username = "testuser"
    test_password = "SecurePass123"
    
    success, message, user_data = auth_service.register_user(test_username, test_password)
    if success:
        print(f"    ✓ 注册成功: {message}")
        print(f"    用户ID: {user_data['user_id']}, 用户名: {user_data['username']}")
    else:
        print(f"    ⚠ 注册提示: {message}")
        # 用户已存在，继续测试
    
    # 测试重复注册
    print("\n[3] 测试重复注册（应失败）...")
    success, message, _ = auth_service.register_user(test_username, test_password)
    if not success:
        print(f"    ✓ 正确拒绝重复注册: {message}")
    else:
        print(f"    ✗ 错误：不应允许重复注册")
    
    # 测试用户登录
    print("\n[4] 测试用户登录...")
    success, message, auth_data = auth_service.login(test_username, test_password)
    if success:
        print(f"    ✓ 登录成功: {message}")
        print(f"    用户ID: {auth_data['user_id']}")
        print(f"    Token过期时间: {auth_data['expires_in']}秒")
        
        # 验证 Token
        print("\n[5] 验证 JWT Token...")
        access_token = auth_data['access_token']
        valid, user_info = auth_service.verify_token(access_token)
        if valid:
            print(f"    ✓ Token 有效")
            print(f"    用户ID: {user_info['user_id']}, 用户名: {user_info['username']}")
        else:
            print(f"    ✗ Token 无效")
        
        # 测试 Token 黑名单（登出）
        print("\n[6] 测试用户登出...")
        success, message = auth_service.logout(access_token)
        if success:
            print(f"    ✓ 登出成功: {message}")
        else:
            print(f"    ✗ 登出失败: {message}")
        
        # 验证 Token 已失效
        print("\n[7] 验证 Token 已失效...")
        valid, _ = auth_service.verify_token(access_token)
        if not valid:
            print(f"    ✓ Token 已失效（在黑名单中）")
        else:
            print(f"    ✗ Token 仍有效（错误）")
        
        # 测试刷新令牌
        print("\n[8] 测试刷新令牌...")
        refresh_token = auth_data['refresh_token']
        success, message, token_data = auth_service.refresh_token(refresh_token)
        if success:
            print(f"    ✓ 令牌刷新成功")
            print(f"    新Token过期时间: {token_data['expires_in']}秒")
        else:
            print(f"    ✗ 令牌刷新失败: {message}")
    else:
        print(f"    ✗ 登录失败: {message}")
    
    # 测试错误密码
    print("\n[9] 测试错误密码...")
    success, message, _ = auth_service.login(test_username, "wrongpassword")
    if not success:
        print(f"    ✓ 正确拒绝错误密码: {message}")
    else:
        print(f"    ✗ 错误：不应允许错误密码登录")
    
    # 测试密码强度验证
    print("\n[10] 测试密码强度验证...")
    success, message, _ = auth_service.register_user("weakuser", "123")
    if not success:
        print(f"    ✓ 正确拒绝弱密码: {message}")
    else:
        print(f"    ✗ 错误：不应允许弱密码")
    
    # 测试用户名验证
    print("\n[11] 测试用户名验证...")
    success, message, _ = auth_service.register_user("ab", "SecurePass123")
    if not success:
        print(f"    ✓ 正确拒绝短用户名: {message}")
    else:
        print(f"    ✗ 错误：不应允许短用户名")
    
    print("\n" + "=" * 60)
    print("验证完成")
    print("=" * 60)


if __name__ == '__main__':
    test_register_and_login()