"""Provision a DocuMind EC2 instance (t3.micro, Ubuntu 22.04, us-east-2).

Usage:
    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...
    python scripts/provision_ec2.py

Outputs the public IP and SSH command when done.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-2")
KEY_NAME = "documind-key"
KEY_PATH = Path.home() / ".ssh" / "documind-key.pem"
SG_NAME = "documind-sg"
INSTANCE_TYPE = "t3.micro"
APP_DIR = "/home/ubuntu/documind"

ec2 = boto3.client("ec2", region_name=REGION)
ec2_res = boto3.resource("ec2", region_name=REGION)


def latest_ubuntu_ami() -> str:
    """Resolve the latest Ubuntu 22.04 LTS AMI in the target region."""
    resp = ec2.describe_images(
        Owners=["099720109477"],  # Canonical
        Filters=[
            {"Name": "name", "Values": ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]},
            {"Name": "state", "Values": ["available"]},
            {"Name": "architecture", "Values": ["x86_64"]},
        ],
    )
    images = sorted(resp["Images"], key=lambda x: x["CreationDate"], reverse=True)
    if not images:
        raise RuntimeError("No Ubuntu 22.04 AMI found in region " + REGION)
    ami = images[0]["ImageId"]
    print(f"  AMI: {ami} ({images[0]['Name'][:60]})")
    return ami


def ensure_key_pair() -> str:
    """Create key pair if it doesn't exist; save .pem locally."""
    if KEY_PATH.exists():
        print(f"  Key pair already exists at {KEY_PATH}")
        return KEY_NAME

    try:
        ec2.describe_key_pairs(KeyNames=[KEY_NAME])
        print(f"  Key pair '{KEY_NAME}' exists in AWS but .pem not found locally.")
        print(f"  Delete it in AWS Console and re-run, or copy your .pem to {KEY_PATH}")
        sys.exit(1)
    except ClientError as e:
        if "InvalidKeyPair.NotFound" not in str(e):
            raise

    resp = ec2.create_key_pair(KeyName=KEY_NAME)
    KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    KEY_PATH.write_text(resp["KeyMaterial"])
    KEY_PATH.chmod(0o400)
    print(f"  Created key pair '{KEY_NAME}' → {KEY_PATH}")
    return KEY_NAME


def ensure_security_group() -> str:
    """Create security group if it doesn't exist; return group ID."""
    try:
        resp = ec2.describe_security_groups(
            Filters=[{"Name": "group-name", "Values": [SG_NAME]}]
        )
        if resp["SecurityGroups"]:
            sg_id = resp["SecurityGroups"][0]["GroupId"]
            print(f"  Security group '{SG_NAME}' exists ({sg_id})")
            return sg_id
    except ClientError:
        pass

    resp = ec2.create_security_group(
        GroupName=SG_NAME,
        Description="DocuMind - SSH, HTTP, FastAPI",
    )
    sg_id = resp["GroupId"]

    ec2.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[
            # SSH
            {"IpProtocol": "tcp", "FromPort": 22, "ToPort": 22,
             "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
            # HTTP (nginx → Streamlit)
            {"IpProtocol": "tcp", "FromPort": 80, "ToPort": 80,
             "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
            # FastAPI direct access
            {"IpProtocol": "tcp", "FromPort": 8888, "ToPort": 8888,
             "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
        ],
    )
    print(f"  Created security group '{SG_NAME}' ({sg_id}): ports 22, 80, 8888")
    return sg_id


def user_data_script() -> str:
    """Cloud-init script that bootstraps the instance on first boot."""
    return """#!/bin/bash
set -euo pipefail
exec > /var/log/documind-init.log 2>&1

echo "=== DocuMind bootstrap start ==="
apt-get update -y
apt-get install -y git curl wget nginx supervisor python3.11 python3.11-venv python3-pip \
    build-essential libpq-dev ca-certificates gnupg lsb-release

# Docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] \
    https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
    > /etc/apt/sources.list.d/docker.list
apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
usermod -aG docker ubuntu
systemctl enable docker

# uv
curl -Ls https://astral.sh/uv/install.sh | bash
export PATH="$HOME/.cargo/bin:$PATH"
ln -sf "$HOME/.cargo/bin/uv" /usr/local/bin/uv

# App dir placeholder — rsync from local fills this in
mkdir -p /home/ubuntu/documind
chown -R ubuntu:ubuntu /home/ubuntu/documind

echo "=== Bootstrap complete — waiting for rsync deploy ==="
"""


def launch_instance(ami: str, sg_id: str) -> dict:
    """Launch the EC2 instance and return instance info dict."""
    resp = ec2.run_instances(
        ImageId=ami,
        InstanceType=INSTANCE_TYPE,
        MinCount=1,
        MaxCount=1,
        KeyName=KEY_NAME,
        SecurityGroupIds=[sg_id],
        UserData=user_data_script(),
        BlockDeviceMappings=[{
            "DeviceName": "/dev/sda1",
            "Ebs": {"VolumeSize": 20, "VolumeType": "gp3", "DeleteOnTermination": True},
        }],
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [
                {"Key": "Name", "Value": "documind"},
                {"Key": "Project", "Value": "DocuMind"},
            ],
        }],
    )
    instance = resp["Instances"][0]
    instance_id = instance["InstanceId"]
    print(f"  Launched {instance_id} ({INSTANCE_TYPE})")
    return instance


def wait_for_running(instance_id: str) -> str:
    """Poll until instance is running and has a public IP. Returns public IP."""
    print("  Waiting for instance to start", end="", flush=True)
    for _ in range(60):
        resp = ec2.describe_instances(InstanceIds=[instance_id])
        inst = resp["Reservations"][0]["Instances"][0]
        state = inst["State"]["Name"]
        if state == "running" and inst.get("PublicIpAddress"):
            print()
            return inst["PublicIpAddress"]
        print(".", end="", flush=True)
        time.sleep(5)
    raise TimeoutError("Instance did not reach running state in 5 minutes")


def save_state(instance_id: str, public_ip: str) -> None:
    state = {"instance_id": instance_id, "public_ip": public_ip, "region": REGION}
    Path("scripts/.ec2_state.json").write_text(json.dumps(state, indent=2))
    print(f"  State saved to scripts/.ec2_state.json")


def main() -> None:
    print("\n=== DocuMind EC2 Provisioner ===\n")

    print("[1/4] Resolving Ubuntu 22.04 AMI…")
    ami = latest_ubuntu_ami()

    print("[2/4] Ensuring key pair…")
    ensure_key_pair()

    print("[3/4] Ensuring security group…")
    sg_id = ensure_security_group()

    print("[4/4] Launching t3.micro instance…")
    instance = launch_instance(ami, sg_id)
    instance_id = instance["InstanceId"]

    public_ip = wait_for_running(instance_id)
    save_state(instance_id, public_ip)

    print(f"""
=== Instance Ready ===

  Instance ID : {instance_id}
  Public IP   : {public_ip}
  Region      : {REGION}

Next step — deploy the app:

  export EC2_IP={public_ip}
  bash scripts/deploy_to_ec2.sh

(Wait ~60s after this for cloud-init to finish before deploying.)
""")


if __name__ == "__main__":
    main()
