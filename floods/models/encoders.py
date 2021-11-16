import timm

# implement here other custom encoders if required
# just a simple wrapper to include custom encoders into the list
available_encoders = {name: timm.create_model for name in timm.list_models()}
