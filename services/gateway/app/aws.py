"""AWS helpers, used only when the gateway runs on AWS (ADR 0004). Locally nothing here runs.

- S3Uploads: the browser uploads images straight to S3 with a short-lived presigned POST,
  because a Lambda request body is limited to 6 MB. Inference then fetches the image with a
  presigned GET URL.
- SigV4: signs calls to the inference and geo Lambda function URLs, which only accept
  requests signed with the gateway's IAM role (so they cannot be called from the internet).
"""

import re
import uuid

import httpx

UPLOAD_KEY = re.compile(r"^uploads/[0-9a-f]{32}\.(tif|tiff|png|jpg|jpeg)$")


class S3Uploads:
    def __init__(self, bucket: str, region: str | None) -> None:
        import boto3
        from botocore.config import Config

        self.bucket = bucket
        # Regional endpoint and SigV4, so presigned URLs work right after the bucket is made.
        self.s3 = boto3.client(
            "s3",
            region_name=region,
            config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
        )

    def new_upload(self, ext: str, max_bytes: int) -> dict:
        key = f"uploads/{uuid.uuid4().hex}{ext}"
        post = self.s3.generate_presigned_post(
            self.bucket,
            key,
            Conditions=[["content-length-range", 1, max_bytes]],
            ExpiresIn=600,
        )
        return {"upload_key": key, "url": post["url"], "fields": post["fields"]}

    def size(self, key: str) -> int | None:
        """Size in bytes of an uploaded object, or None if it does not exist."""
        from botocore.exceptions import ClientError

        try:
            return self.s3.head_object(Bucket=self.bucket, Key=key)["ContentLength"]
        except ClientError:
            return None

    def download_url(self, key: str) -> str:
        return self.s3.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=300
        )


class SigV4(httpx.Auth):
    """httpx auth that signs requests for Lambda function URLs (service "lambda")."""

    requires_request_body = True

    def __init__(self, region: str) -> None:
        import botocore.session

        self.region = region
        self.session = botocore.session.get_session()

    def auth_flow(self, request: httpx.Request):
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        headers = {"host": request.url.netloc.decode()}
        if "content-type" in request.headers:
            headers["content-type"] = request.headers["content-type"]
        aws = AWSRequest(
            method=request.method, url=str(request.url), data=request.content, headers=headers
        )
        SigV4Auth(self.session.get_credentials(), "lambda", self.region).add_auth(aws)
        for name in ("Authorization", "X-Amz-Date", "X-Amz-Security-Token"):
            if name in aws.headers:
                request.headers[name] = aws.headers[name]
        yield request
