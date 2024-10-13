from functools import wraps
from rest_framework.response import Response
from ml_deploy.models import Conversation

def check_conversation_access(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        conversation_id = kwargs.get('conversation_id')
        if conversation_id:
            try:
                conversation = Conversation.objects.get(conversation_id=conversation_id)
                if request.user in conversation.users.all():
                    return view_func(request, *args, **kwargs)
                else:
                    return Response({'error': 'You dont have access to this conversation.'}, status=403)
            except Conversation.DoesNotExist:
                return Response({'error': 'Conversation does not exist.'}, status=404)
        else:
            return Response({'error': 'Conversation ID is missing.'}, status=400)
    return wrapper