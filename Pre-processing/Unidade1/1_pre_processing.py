from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()
print(volunteer.shape)

print(volunteer.info())

