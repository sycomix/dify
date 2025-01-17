from flask_login import current_user
from libs.login import login_required
from flask_restful import Resource
from functools import wraps

from werkzeug.exceptions import NotFound

from controllers.console.wraps import account_initialization_required
from extensions.ext_database import db
from models.model import InstalledApp


def installed_app_required(view=None):
    def decorator(view):
        @wraps(view)
        def decorated(*args, **kwargs):
            if not kwargs.get('installed_app_id'):
                raise ValueError('missing installed_app_id in path parameters')

            installed_app_id = kwargs.get('installed_app_id')
            installed_app_id = str(installed_app_id)

            del kwargs['installed_app_id']

            installed_app = (
                db.session.query(InstalledApp)
                .filter(
                    InstalledApp.id == installed_app_id,
                    InstalledApp.tenant_id == current_user.current_tenant_id,
                )
                .first()
            )

            if installed_app is None:
                raise NotFound('Installed app not found')

            if not installed_app.app:
                db.session.delete(installed_app)
                db.session.commit()

                raise NotFound('Installed app not found')

            return view(installed_app, *args, **kwargs)

        return decorated

    return decorator(view) if view else decorator


class InstalledAppResource(Resource):
    # must be reversed if there are multiple decorators
    method_decorators = [installed_app_required, account_initialization_required, login_required]
