def app(scope, receive, send):
    assert scope["type"] == "http"
    
    async def asgi():
        await send({"type": "http.response.start", "status": 200})
        await send({"type": "http.response.body", "body": b"Hello AI DevLab OS!"})
    
    return asgi()
