from huggingface_hub import HfApi, login
login()
api = HfApi()
models = api.list_models(author='IDEA-CCNL', sort='downloads', direction=-1)
downloads = 0
likes = 0
for model in models:
    downloads += model.downloads
    likes += model.likes
    created_at = api.list_repo_commits(model.modelId)[-1].created_at
    print(f"{model.modelId}:{model.downloads}:{model.likes}")
print(downloads, likes)
