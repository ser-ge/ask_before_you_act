import base64
import glob
import io
from IPython.display import HTML
from gym.wrappers import Monitor
from IPython import display
from IPython.core.display import HTML
from IPython.display import Javascript
import json
from ipywidgets import interact, widgets
import matplotlib.pyplot as plt


def render_episode():
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")


def wrap_env_video_monitor(env):
    env = Monitor(env, './video', force=True)
    env.metadata['video.frames_per_second'] = 1
    env.metadata['video.output_frames_per_second'] = 1
    return env


def render_qa(questions, answers):
    display.display(
        HTML(
            """
          <style>
            video::-webkit-media-controls {
            display: none;
          }
        
          .imessage {
          border-radius: 0.25rem;
          display: flex;
          flex-direction: column;
          font-family: "SanFrancisco";
          font-size: 1.1rem;
          /* margin: 0 auto 1rem; */
          max-width: 230px;
          padding: 0.5rem 1.5rem;
        }
        
        .imessage p {
          border-radius: 1.15rem;
          line-height: 1.25;
          max-width: 75%;
          padding: 0.5rem .875rem !important;
          position: relative;
          word-wrap: break-word;
        }
        
        .imessage p::before,
        .imessage p::after {
          bottom: -0.1rem;
          content: "";
          height: 1rem;
          position: absolute;
        }
        
        p.from-me {
          align-self: flex-end;
          background-color: #248bf5;
          color: #fff; 
        }
        
        p.from-me::before {
          border-bottom-left-radius: 0.8rem 0.7rem;
          border-right: 1rem solid #248bf5;
          right: -0.35rem;
          transform: translate(0, -0.1rem);
        }
        
        p.from-me::after {
          background-color: #383838;
          border-bottom-left-radius: 0.5rem;
          right: -40px;
          transform:translate(-30px, -2px);
          width: 10px;
        }
        
        p[class^="from-"] {
          margin: 0.5rem 0;
          width: fit-content;
        }
        
        p:first-letter{
          text-transform: capitalize
        }
        
        p.from-me ~ p.from-me {
          margin: 0.25rem 0 0;
        }
        
        p.from-me ~ p.from-me:not(:last-child) {
          margin: 0.25rem 0 0;
        }
        
        p.from-me ~ p.from-me:last-child {
          margin-bottom: 0.5rem;
        }
        
        p.from-them {
          align-items: flex-start;
          background-color: #e5e5ea;
          color: #000;
        }
        
        p.from-them:before {
          border-bottom-right-radius: 0.8rem 0.7rem;
          border-left: 1rem solid #e5e5ea;
          left: -0.35rem;
          transform: translate(0, -0.1rem);
        }
        
        p.from-them::after {
          background-color: #383838;
          border-bottom-right-radius: 0.5rem;
          left: 20px;
          transform: translate(-30px, -2px);
          width: 10px;
        }
        
        .no-tail::before {
          display: none;
        }
        
        /* general styling */
        @font-face {
          font-family: "SanFrancisco";
          src:
            url("https://cdn.rawgit.com/AllThingsSmitty/fonts/25983b71/SanFrancisco/sanfranciscodisplay-regular-webfont.woff2") format("woff2"),
            url("https://cdn.rawgit.com/AllThingsSmitty/fonts/25983b71/SanFrancisco/sanfranciscodisplay-regular-webfont.woff") format("woff");
        }
        
        body {  
          font-family: -apple-system, 
            BlinkMacSystemFont, 
            "Segoe UI", 
            Roboto, 
            Oxygen-Sans, 
            Ubuntu, 
            Cantarell, 
            "Helvetica Neue", 
            sans-serif;
          font-weight: normal;
          margin: 0;
        }
        
        @media screen and (max-width: 800px) {
          body {
            margin: 0 0.5rem;
          }
        
          .imessage {
            font-size: 1.05rem;
            margin: 0 auto 1rem;
            max-width: 600px;
            padding: 0.25rem 0.875rem;
          }
        
          .imessage p {
            margin: 0.5rem 0;
          }
        
        }
        
        .container {
                width: 100%;
                overflow-y: scroll;
                /* padding-top: 20px;
                padding-left: 40px; */
                height: 400px;
                scrollbar-color: black;
              }
          </style>
  """))

    display.display(Javascript('''
      async function run() {
        qa = (%s)
        answers = qa[1]
        questions = qa[0]
        area = document.querySelector("#output-body .output_subarea")
        area.style.display='flex'
    
        while (true) {
          let container = document.createElement("div");
          let messages = document.createElement("div");
          let empty = document.createElement("div");
          area.appendChild(container);
          container.appendChild(messages);
          container.classList.add("container");
          empty.style.height = '100px'
          empty.style.width = '100%%'
          container.appendChild(empty);
    
          messages.classList.add("imessage")
    
          for (let i = 0; i < answers.length; i++) {
              let element = document.createElement("p");
              // element.innerText = (i+1) + '. ' + questions[i] +'?'
              element.innerText = questions[i] +'?'
              element.className = "from-me"
              messages.appendChild(element);
    
              await new Promise(resolve => setTimeout(resolve, 300));
    
              let element2 = document.createElement("p");
              element2.innerText = answers[i]
              element2.className = "from-them"
              messages.appendChild(element2);
              await new Promise(resolve => setTimeout(resolve, 700));
    
              container.scrollTop = container.scrollHeight - container.clientHeight;
          }
          await new Promise(resolve => setTimeout(resolve, 1000));
          container.remove()
        }
      }
      run()
    ''' % json.dumps([questions, answers])))


def show_question_input(env):
    question_input = widgets.Text(
        value='Green goal is north?',
        placeholder='Green goal is north?',
        description='Question:',
        disabled=False
    )
    display.display(question_input)

    def on_ask(wdgt):
        question = wdgt.value
        question_n = question.replace('?', '').lower()
        display.display(wdgt.value)
        answer, reward_qa = env.answer(question_n)
        display.display(str(answer))

    question_input.on_submit(on_ask)


def render_env(env, step=0):
    plt.figure(1, figsize=(8, 8))
    plt.clf()
    plt.axis('off')
    plt.imshow(env.render(mode='rgb_array'))
    # pause for plots to update
    display.display(plt.gcf())
    display.clear_output(wait=True)
    plt.pause(0.0001)

    display.display(
        HTML(
            """
        <style>
          .output_image>img {
            background: inherit;
          }
          </style>
      """))
