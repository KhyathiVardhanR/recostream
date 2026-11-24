# AWS deployment: an S3 bucket for versioned model artifacts and one EC2
# instance that runs the docker-compose stack. The instance reads models
# through an IAM role (no keys on the box).
#
#   cd deploy/terraform
#   terraform init
#   terraform apply -var key_name=<your-ec2-keypair> -var allowed_cidr=<your-ip>/32
#   # then: scripts/deploy_ec2.sh $(terraform output -raw public_ip)

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws    = { source = "hashicorp/aws", version = "~> 5.60" }
    random = { source = "hashicorp/random", version = "~> 3.6" }
  }
}

variable "region" {
  type    = string
  default = "us-east-1"
}
variable "instance_type" {
  type    = string
  default = "c6i.2xlarge"
}
variable "key_name" {
  type = string
}
variable "allowed_cidr" {
  type        = string
  description = "CIDR allowed to reach SSH, the API and Prometheus"
}
variable "bucket_name" {
  type    = string
  default = ""
}

provider "aws" {
  region = var.region
}

resource "random_id" "suffix" {
  byte_length = 4
}

locals {
  bucket = var.bucket_name != "" ? var.bucket_name : "recostream-models-${random_id.suffix.hex}"
}

resource "aws_s3_bucket" "models" {
  bucket        = local.bucket
  force_destroy = true
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket                  = aws_s3_bucket.models.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "app" {
  name               = "recostream-app-${random_id.suffix.hex}"
  assume_role_policy = data.aws_iam_policy_document.assume.json
}

data "aws_iam_policy_document" "models_rw" {
  statement {
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.models.arn]
  }
  statement {
    actions   = ["s3:GetObject", "s3:PutObject"]
    resources = ["${aws_s3_bucket.models.arn}/*"]
  }
}

resource "aws_iam_role_policy" "models_rw" {
  role   = aws_iam_role.app.id
  policy = data.aws_iam_policy_document.models_rw.json
}

resource "aws_iam_instance_profile" "app" {
  name = "recostream-app-${random_id.suffix.hex}"
  role = aws_iam_role.app.name
}

data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }
}

resource "aws_security_group" "app" {
  name = "recostream-${random_id.suffix.hex}"
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_cidr]
  }
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = [var.allowed_cidr]
  }
  ingress {
    from_port   = 9090
    to_port     = 9090
    protocol    = "tcp"
    cidr_blocks = [var.allowed_cidr]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "app" {
  ami                    = data.aws_ami.al2023.id
  instance_type          = var.instance_type
  key_name               = var.key_name
  iam_instance_profile   = aws_iam_instance_profile.app.name
  vpc_security_group_ids = [aws_security_group.app.id]
  root_block_device {
    volume_size = 60
    volume_type = "gp3"
  }
  user_data = <<-EOF
    #!/bin/bash
    dnf install -y docker git
    systemctl enable --now docker
    mkdir -p /usr/local/lib/docker/cli-plugins
    curl -sSL https://github.com/docker/compose/releases/download/v2.29.2/docker-compose-linux-x86_64 \
      -o /usr/local/lib/docker/cli-plugins/docker-compose
    chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
    usermod -aG docker ec2-user
    echo "S3_BUCKET=${local.bucket}" >> /etc/environment
    echo "AWS_REGION=${var.region}" >> /etc/environment
  EOF
  tags = { Name = "recostream" }
}

output "public_ip" {
  value = aws_instance.app.public_ip
}
output "bucket" {
  value = aws_s3_bucket.models.bucket
}
