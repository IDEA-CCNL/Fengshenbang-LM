from huggingface_hub import HfApi, login, ModelFilter
login()
api = HfApi()
fs_filter = ModelFilter(author='IDEA-CCNL')
models = api.list_models(filter=fs_filter, sort='downloads', direction=-1)
downloads = 0
likes = 0
for model in models:
    downloads += model.downloads
    likes += model.likes
    created_at = api.list_repo_commits(model.modelId)[-1].created_at
    print(f"{model.modelId}:{model.downloads}:{model.likes}")
print(downloads, likes)