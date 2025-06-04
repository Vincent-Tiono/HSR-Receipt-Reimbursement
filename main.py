# import uvicorn

# if __name__ == "__main__":
#     uvicorn.run("ticket_extractor.api:app", host="0.0.0.0", port=57543, reload=True)

import uvicorn
from ticket_extractor.api import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)