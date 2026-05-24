"""Provision EC2 t2.micro for DocuMind FastAPI + Streamlit.

FREE TIER: t2.micro, 750 hrs/month.
Installs: FastAPI backend on port 8000, Streamlit UI on port 8501.
nginx reverse proxy: / → FastAPI, /ui → Streamlit.

Run: python aws/ec2/setup_ec2.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from aws.config import REGION, ACCOUNT_ID, IAM_ROLE_NAME

console = Console()

ec2 = boto3.client("ec2", region_name=REGION)
ec2_resource = boto3.resource("ec2", region_name=REGION)

INSTANCE_NAME = "documind-backend"
KEY_NAME = "documind-key"
KEY_FILE = os.path.expanduser("~/.ssh/documind-key.pem")
SG_NAME = "documind-backend-sg"


def get_local_ip() -> str:
    """Get the current machine's public IP for SSH security group."""
    import urllib.request
    try:
        return urllib.request.urlopen("https://checkip.amazonaws.com", timeout=5).read().decode().strip()
    except Exception:
        return "0.0.0.0"


def ensure_key_pair() -> str:
    """Create key pair if not exists. Returns key name."""
    if os.path.exists(KEY_FILE):
        console.print(f"  [yellow]~[/yellow] Key pair exists: {KEY_FILE}")
        return KEY_NAME
    try:
        resp = ec2.create_key_pair(KeyName=KEY_NAME, KeyType="rsa", KeyFormat="pem")
        key_material = resp["KeyMaterial"]
        os.makedirs(os.path.dirname(KEY_FILE), exist_ok=True)
        with open(KEY_FILE, "w") as f:
            f.write(key_material)
        os.chmod(KEY_FILE, 0o400)
        console.print(f"  [green]✓[/green] Key pair created: {KEY_FILE}")
    except ClientError as e:
        if "InvalidKeyPair.Duplicate" in str(e):
            console.print(f"  [yellow]~[/yellow] Key pair {KEY_NAME} exists in AWS (missing local .pem)")
        else:
            raise
    return KEY_NAME


def ensure_security_group() -> str:
    """Create or return existing security group ID."""
    try:
        vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])
        vpc_id = vpcs["Vpcs"][0]["VpcId"]
    except (KeyError, IndexError):
        vpc_id = None

    try:
        sgs = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [SG_NAME]}])
        if sgs["SecurityGroups"]:
            sg_id = sgs["SecurityGroups"][0]["GroupId"]
            console.print(f"  [yellow]~[/yellow] Security group exists: {sg_id}")
            return sg_id
    except ClientError:
        pass

    kwargs = {
        "GroupName": SG_NAME,
        "Description": "DocuMind backend: SSH + HTTP + FastAPI",
    }
    if vpc_id:
        kwargs["VpcId"] = vpc_id

    sg = ec2.create_security_group(**kwargs)
    sg_id = sg["GroupId"]

    local_ip = get_local_ip()
    ec2.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[
            {"IpProtocol": "tcp", "FromPort": 22, "ToPort": 22,
             "IpRanges": [{"CidrIp": f"{local_ip}/32", "Description": "SSH from my IP"}]},
            {"IpProtocol": "tcp", "FromPort": 80, "ToPort": 80,
             "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
            {"IpProtocol": "tcp", "FromPort": 8000, "ToPort": 8000,
             "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "FastAPI"}]},
            {"IpProtocol": "tcp", "FromPort": 8501, "ToPort": 8501,
             "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "Streamlit"}]},
        ],
    )
    console.print(f"  [green]✓[/green] Security group created: {sg_id}")
    return sg_id


def get_amazon_linux_ami() -> str:
    """Get latest Amazon Linux 2023 AMI ID for the region."""
    resp = ec2.describe_images(
        Owners=["amazon"],
        Filters=[
            {"Name": "name", "Values": ["al2023-ami-*-x86_64"]},
            {"Name": "state", "Values": ["available"]},
        ],
    )
    images = sorted(resp["Images"], key=lambda x: x["CreationDate"], reverse=True)
    if not images:
        raise RuntimeError("No Amazon Linux 2023 AMI found in this region")
    return images[0]["ImageId"]


def launch_instance(sg_id: str, key_name: str) -> dict:
    """Launch t2.micro instance with bootstrap script."""
    bootstrap_path = Path(__file__).parent / "bootstrap.sh"
    user_data = bootstrap_path.read_text() if bootstrap_path.exists() else "#!/bin/bash\necho 'bootstrap missing'"

    ami_id = get_amazon_linux_ami()
    console.print(f"  AMI: {ami_id}")

    resp = ec2.run_instances(
        ImageId=ami_id,
        InstanceType="t2.micro",  # FREE TIER
        KeyName=key_name,
        SecurityGroupIds=[sg_id],
        MinCount=1,
        MaxCount=1,
        UserData=user_data,
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [
                {"Key": "Name", "Value": INSTANCE_NAME},
                {"Key": "Project", "Value": "DocuMind"},
            ],
        }],
        IamInstanceProfile={"Name": IAM_ROLE_NAME} if IAM_ROLE_NAME else {},
    )
    instance = resp["Instances"][0]
    console.print(f"  [green]✓[/green] Instance launched: {instance['InstanceId']}")
    return instance


def allocate_elastic_ip(instance_id: str) -> str:
    """Allocate and associate an Elastic IP."""
    alloc = ec2.allocate_address(Domain="vpc")
    allocation_id = alloc["AllocationId"]
    public_ip = alloc["PublicIp"]

    # Wait for instance to be running
    console.print(f"  Waiting for instance {instance_id} to be running …")
    waiter = ec2.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id])

    ec2.associate_address(InstanceId=instance_id, AllocationId=allocation_id)
    console.print(f"  [green]✓[/green] Elastic IP: {public_ip}")
    return public_ip


def main() -> None:
    console.print("\n[bold cyan]DocuMind EC2 Setup (t2.micro FREE TIER)[/bold cyan]\n")

    # Check if instance already exists
    existing = ec2.describe_instances(Filters=[
        {"Name": "tag:Name", "Values": [INSTANCE_NAME]},
        {"Name": "instance-state-name", "Values": ["running", "stopped", "pending"]},
    ])
    if existing["Reservations"]:
        inst = existing["Reservations"][0]["Instances"][0]
        ip = inst.get("PublicIpAddress", "no IP yet")
        console.print(f"[yellow]Instance already exists: {inst['InstanceId']} ({ip})[/yellow]")
        console.print(f"  SSH: ssh -i {KEY_FILE} ec2-user@{ip}")
        console.print(f"  API: http://{ip}/health")
        console.print(f"  UI:  http://{ip}/ui")
        return

    key_name = ensure_key_pair()
    sg_id = ensure_security_group()
    instance = launch_instance(sg_id, key_name)
    instance_id = instance["InstanceId"]
    public_ip = allocate_elastic_ip(instance_id)

    console.print(f"\n[bold green]EC2 instance ready![/bold green]")
    console.print(f"  Instance ID: {instance_id}")
    console.print(f"  Public IP:   {public_ip}")
    console.print(f"\n[yellow]Bootstrap running — wait ~5 min for API to be ready.[/yellow]")
    console.print(f"\n  SSH:  ssh -i {KEY_FILE} ec2-user@{public_ip}")
    console.print(f"  API:  http://{public_ip}/health  (after bootstrap)")
    console.print(f"  UI:   http://{public_ip}/ui       (after bootstrap)")
    console.print(f"\n  Set in .env: DOCUMIND_EC2_IP={public_ip}")


if __name__ == "__main__":
    main()
