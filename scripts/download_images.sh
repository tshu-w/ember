wget --output-document data/ali/images.tgz https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBbENwU280NzBXSXlvcGs2WDNjZ1RIUFBtUjZhUHc/root/content
tar -xvf data/ali/images.tgz --directory=data/ali # --use-compress-program=pigz
