#!/usr/bin/env python3
"""
生成自签名 SSL 证书
用于开发和测试 HTTPS 功能
"""

import os
from pathlib import Path
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import datetime
import ipaddress


def generate_self_signed_cert(cert_dir='certs', days_valid=365):
    """
    生成自签名 SSL 证书
    
    Args:
        cert_dir: 证书保存目录
        days_valid: 证书有效天数
    """
    # 创建证书目录
    cert_path = Path(cert_dir)
    cert_path.mkdir(parents=True, exist_ok=True)
    
    cert_file = cert_path / 'server.crt'
    key_file = cert_path / 'server.key'
    
    # 如果证书已存在，询问是否覆盖
    if cert_file.exists() or key_file.exists():
        print(f"⚠️  证书文件已存在:")
        if cert_file.exists():
            print(f"   - {cert_file}")
        if key_file.exists():
            print(f"   - {key_file}")
        
        response = input("\n是否覆盖现有证书？(y/N): ").strip().lower()
        if response != 'y':
            print("❌ 取消生成")
            return False
    
    print(f"🔐 正在生成自签名 SSL 证书...")
    
    # 生成私钥
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    
    # 创建证书信息
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, u"CN"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, u"Beijing"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, u"Beijing"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"PostureAnalysisM"),
        x509.NameAttribute(NameOID.COMMON_NAME, u"localhost"),
    ])
    
    # 创建证书
    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        private_key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.datetime.utcnow()
    ).not_valid_after(
        datetime.datetime.utcnow() + datetime.timedelta(days=days_valid)
    ).add_extension(
        x509.SubjectAlternativeName([
            x509.DNSName(u"localhost"),
            x509.IPAddress(ipaddress.IPv4Address(u"127.0.0.1")),
            x509.IPAddress(ipaddress.IPv4Address(u"0.0.0.0")),
        ]),
        critical=False,
    ).sign(private_key, hashes.SHA256())
    
    # 保存私钥
    with open(key_file, 'wb') as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))
    print(f"✅ 私钥已保存: {key_file}")
    
    # 保存证书
    with open(cert_file, 'wb') as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    print(f"✅ 证书已保存: {cert_file}")
    
    # 设置文件权限（仅Unix系统）
    try:
        os.chmod(key_file, 0o600)  # 仅所有者可读写
        os.chmod(cert_file, 0o644)  # 所有者可读写，其他人只读
    except:
        pass
    
    print(f"\n🎉 SSL 证书生成成功！")
    print(f"   证书有效期: {days_valid} 天")
    print(f"   证书路径: {cert_file}")
    print(f"   私钥路径: {key_file}")
    print(f"\n📝 使用说明:")
    print(f"   1. 在 config.yaml 中设置 ssl.enable: true")
    print(f"   2. 启动服务器: python -m src.socketio_server")
    print(f"   3. 浏览器访问: https://localhost:8443")
    print(f"\n⚠️  注意:")
    print(f"   - 这是自签名证书，浏览器会显示安全警告")
    print(f"   - 在浏览器中点击'高级'→'继续访问'即可")
    print(f"   - 生产环境请使用正式的 SSL 证书")
    
    return True


if __name__ == '__main__':
    import sys
    
    # 支持命令行参数
    cert_dir = 'certs'
    days_valid = 365
    
    if len(sys.argv) > 1:
        cert_dir = sys.argv[1]
    if len(sys.argv) > 2:
        days_valid = int(sys.argv[2])
    
    try:
        success = generate_self_signed_cert(cert_dir, days_valid)
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
