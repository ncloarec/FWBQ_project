%% tstextbook.cls
%% Copyright 2016 Typesetters.se
%
% This work may be distributed and/or modified under the
% conditions of the LaTeX Project Public License, either version 1.3
% of this license or (at your option) any later version.
% The latest version of this license is in
%   http://www.latex-project.org/lppl.txt
% and version 1.3 or later is part of all distributions of LaTeX
% version 2005/12/01 or later.
%
% This is a free LaTeX text book template from Typesetters.se.
% See tstextbook.tex for an example of how to use this class.

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{beautiful_article}[2017/02/19 Free LaTeX text book template from Typesetters.se]
\LoadClass[11pt,onecolumn,titlepage,twoside,openany]{book}

% Creative Common License
%--------------------------------------------------------------------------

% Language
%--------------------------------------------------------------------------
%\RequirePackage[swedish]{babel}
\usepackage{polyglossia}
\setmainlanguage{english}

% Geometry
%--------------------------------------------------------------------------
% \RequirePackage[top=2.5cm,bottom=2.5cm,left=3cm,right=3cm,
%                 marginparwidth=1.1cm,marginparsep=2mm]{geometry}
\RequirePackage[hmargin=1.5cm, vmargin=1.5cm]{geometry}
% Chapter Epigraph
%--------------------------------------------------------------------------
\usepackage{epigraph}
\usepackage{varwidth}

\renewcommand{\epigraphsize}{\small}
\setlength{\epigraphwidth}{0.6\textwidth}
\renewcommand{\textflush}{flushright}
\renewcommand{\sourceflush}{flushright}
% A useful addition
\newcommand{\epitextfont}{\itshape}
\newcommand{\episourcefont}{\scshape}

\makeatletter
\newsavebox{\epi@textbox}
\newsavebox{\epi@sourcebox}
\newlength\epi@finalwidth
\renewcommand{\epigraph}[2]{%
  \vspace{\beforeepigraphskip}
  {\epigraphsize\begin{\epigraphflush}
   \epi@finalwidth=\z@
   \sbox\epi@textbox{%
     \varwidth{\epigraphwidth}
     \begin{\textflush}\epitextfont#1\end{\textflush}
     \endvarwidth
   }%
   \epi@finalwidth=\wd\epi@textbox
   \sbox\epi@sourcebox{%
     \varwidth{\epigraphwidth}
     \begin{\sourceflush}\episourcefont#2\end{\sourceflush}%
     \endvarwidth
   }%
   \ifdim\wd\epi@sourcebox>\epi@finalwidth
     \epi@finalwidth=\wd\epi@sourcebox
   \fi
   \leavevmode\vbox{
     \hb@xt@\epi@finalwidth{\hfil\box\epi@textbox}
     \vskip1.75ex
     \hrule height \epigraphrule
     \vskip.75ex
     \hb@xt@\epi@finalwidth{\hfil\box\epi@sourcebox}
   }%
   \end{\epigraphflush}
   \vspace{\afterepigraphskip}}}
\makeatother

% Fonts
%--------------------------------------------------------------------------
% \RequirePackage{kpfonts}
%
% \usepackage{fontspec}
% \setmainfont{Georgia}
% \setsansfont{Trebuchet MS}
% \setmonofont{Georgia}
%
\usepackage{amsfonts}
\usepackage[bitstream-charter]{mathdesign}% la police des beaux cours d'autom'

% Colors
%--------------------------------------------------------------------------
\RequirePackage{xcolor}
\definecolor{tssteelblue}{RGB}{70,130,180}
\definecolor{tsorange}{RGB}{255,138,88}
\definecolor{tsblue}{RGB}{23,74,117}
\definecolor{tsforestgreen}{RGB}{21,122,81}
\definecolor{tsyellow}{RGB}{255,185,88}
\definecolor{tsgrey}{RGB}{200,200,200}
\definecolor{ggred}{RGB}{219,68,55}
\definecolor{ggblue}{RGB}{66,133,244}
\definecolor{ggyellow}{RGB}{244,160,0}
\definecolor{gggreen}{RGB}{15,157,88}
\definecolor{plum}{rgb}{0.56, 0.27, 0.52}
\definecolor{midnightblue}{RGB}{0,103,148}
\definecolor{royalblue}{rgb}{0.25, 0.41, 0.88}
\definecolor{periwinkle}{rgb}{0.8, 0.8, 1.0}
\definecolor{mypurple}{RGB}{128, 0, 128}
\definecolor{electriccrimson}{rgb}{1.0, 0.0, 0.25}
\definecolor{myorange}{RGB}{255, 145, 0}

% Graphics
%--------------------------------------------------------------------------
\RequirePackage{graphicx}
\graphicspath{{images/}}
\RequirePackage{tikz}
\RequirePackage[pages=some]{background}
\backgroundsetup{scale=1,color=black,opacity=1.0,angle=0,
contents={\includegraphics[width=\paperwidth,height=\paperheight]{cover}}}
\RequirePackage{wrapfig}

% Header and footer styling
%--------------------------------------------------------------------------
\RequirePackage{fancyhdr}
\setlength{\headheight}{15.0pt}
\fancyhf{}
\fancyhead[LE,RO]{\bfseries\thepage}
\fancyhead[RE]{\it\sf\nouppercase{\rightmark}}
\fancyhead[LO]{\it\sf\nouppercase{\leftmark}}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

% Section styling
%--------------------------------------------------------------------------
% \RequirePackage{sectsty}
% \sectionfont{\sf\color{tssteelblue}\sectionrule{0ex}{0pt}{-1ex}{1pt}}
\renewcommand{\thesection}{\arabic{section}}

% code styling
%--------------------------------------------------------------------------
\usepackage{listings}
\lstloadlanguages{Mathematica}

% algoritms styling
%--------------------------------------------------------------------------
\usepackage[german,onelanguage]{algorithm2e}


% Chapter styling
%--------------------------------------------------------------------------
\makeatletter

% Numbered chapters (with minitoc)
\renewcommand{\@makechapterhead}[1]{
  \begin{tikzpicture}[remember picture,overlay]
    \node at (current page.north west)
  {
    \begin{tikzpicture}[remember picture,overlay]
      % Background image
      \node[anchor=north west,inner sep=0pt] at (0,0)
           {\includegraphics[width=\paperwidth]{background}};
      % Chapter heading
      \draw[anchor=west, inner sep=0pt] (-0.1cm,-1.5cm) node
           [black,
            ultra thick,
            fill=tsorange!10,
            fill opacity=.6,
            inner sep=10pt]
           (0,0)
           {\parbox[t][1.6cm][t]{\paperwidth}
           {\huge\bfseries\sffamily\flushleft\thechapter. #1}};
      % Chapter contents
      \draw[anchor=west,inner sep=0pt] (8.8cm,-7cm) node
           [tssteelblue,
            ultra thick,
            fill=white,
            fill opacity=.8,
            draw=tssteelblue,
            draw opacity=1,
            line width=1.5pt,
            rounded corners,
            inner sep=0pt]
           (0,0)
           {\hspace{1cm}\parbox[t][7.1cm][t]{11cm}
           {\vspace{-1cm}\huge\bfseries\sffamily\flushleft
            \textcolor{black}{\sffamily\minitoc}}};
    \end{tikzpicture}
  };
\end{tikzpicture}
\vspace{7.5cm}
}

% Unnumbered chapters (without minitoc)
\renewcommand{\@makeschapterhead}[1]{
  \begin{tikzpicture}[remember picture,overlay]
    \node at (current page.north west)
  {
    \begin{tikzpicture}[remember picture,overlay]
      % Background image
      \node[anchor=north west,inner sep=0pt] at (0,0)
           {\includegraphics[width=\paperwidth]{background}};
      % Chapter heading
      \draw[anchor=west,inner sep=0pt] (-0.1cm,-1.5cm) node
           [black,
            ultra thick,
            fill=tsorange!10,
            fill opacity=.6,
            inner sep=10pt]
           (0,0)
           {\parbox[t][1.6cm][t]{\paperwidth}
           {\huge\bfseries\sffamily\flushleft #1}};
    \end{tikzpicture}
  };
\end{tikzpicture}
\vspace{8cm}
}

% Minitoc styling
%--------------------------------------------------------------------------
\RequirePackage{minitoc}
\setcounter{minitocdepth}{2}
\dominitoc[n]
\nomtcrule
\nomtcpagenumbers
\renewcommand{\mtifont}{\large\sffamily}
\renewcommand{\mtcfont}{\small\sffamily}
\renewcommand{\mtcSfont}{\small\sffamily}
\renewcommand{\mtcSSfont}{\small\sffamily}
\renewcommand{\mtcSSSfont}{\small\sffamily}

% Table of contents styling
%--------------------------------------------------------------------------
\RequirePackage{titletoc}
\setcounter{tocdepth}{2}

\titlecontents{chapter}[0cm]
{\addvspace{14pt}\large\sffamily\bfseries}
{\color{tssteelblue}\contentslabel[\Large\thecontentslabel]{1.25cm}}{}
{\color{tssteelblue}\normalsize\sffamily\bfseries
 \;\titlerule*[.5pc]{.}\;\thecontentspage}

\titlecontents{section}[0cm]
{\addvspace{6pt}\sffamily\bfseries}
{\contentslabel[\thecontentslabel]{1.25cm}}{}
{\sffamily\hfill\color{black}\thecontentspage}[]



% Boxes for theorems
%--------------------------------------------------------------------------
\newenvironment{boxtheorem}
{
\abovedisplayskip=0pt
\begin{tcolorbox}
[enhanced jigsaw,breakable,pad at break*=1mm,
 colback=black!5,colframe=tsorange]}
{\end{tcolorbox}}

\newenvironment{boxdefinition}
{
\abovedisplayskip=0pt
\begin{tcolorbox}
[enhanced jigsaw,breakable,pad at break*=1mm,
 colback=yellow!10!white,boxrule=0pt,frame hidden,
 borderline west={1.5mm}{-2mm}{tsforestgreen}]}
{\end{tcolorbox}}

\newenvironment{boxcomputation}
{
\abovedisplayskip=0pt
\begin{tcolorbox}
[enhanced jigsaw,breakable,pad at break*=1mm,
 colback=white,boxrule=0pt,frame hidden,
 leftrule=1mm,
 borderline west={1mm}{-2mm}{mypurple}]}
{\end{tcolorbox}}

\newenvironment{boxexercise}
{
\abovedisplayskip=0pt
\begin{tcolorbox}
[enhanced jigsaw,breakable,pad at break*=1mm,
 colback=tsyellow!20,boxrule=0pt,frame hidden]}
{\end{tcolorbox}}

\newenvironment{boxproblem}
{
\abovedisplayskip=0pt
\begin{tcolorbox}
[enhanced jigsaw,breakable,pad at break*=1mm,
 colback=tsorange!25,boxrule=0pt,frame hidden]}
{\end{tcolorbox}}

\newenvironment{boxprogramming}
{
\abovedisplayskip=0pt
\begin{tcolorbox}
[enhanced jigsaw,breakable,pad at break*=1mm,
 colback=tsblue!10,boxrule=0pt,frame hidden]}
{\end{tcolorbox}}

\newenvironment{boxexample}
{
\abovedisplayskip=0pt
\begin{tcolorbox}
[enhanced jigsaw,breakable,pad at break*=0mm,
 colback=tsgrey!20,boxrule=0pt,frame hidden]}
{\end{tcolorbox}}

\newenvironment{mydescription}[1][0cm]
{\begin{description}[font=$\bullet$~\normalfont\bfseries\color{myorange!90!black}, leftmargin=#1]}
{\end{description}}

\newenvironment{myitemize}[1][.5cm]
{\begin{itemize}[font= \color{blue} \large, label= $\bullet$, leftmargin=#1, noitemsep,topsep=0pt]}
{\end{itemize}}

% Styling of theorems
%--------------------------------------------------------------------------
\RequirePackage{amsthm}

\newtheoremstyle{styletheorem}
{0pt}{0pt}{\normalfont}{0pt}{\small\bf\sffamily\color{tsorange}}{\;}{0.25em}
{\small\sffamily\color{tsorange}\thmname{#1}
 \nobreakspace\thmnumber{\@ifnotempty{#1}{}\@upn{#2}}
 \thmnote{\nobreakspace\the\thm@notefont\sffamily\bfseries\color{black} (#3)}}

\newtheoremstyle{styledefinition}
{0pt}{0pt}{\normalfont}{0pt}{\small\bf\sffamily\color{black}}{\;}{0.25em}
{\small\sffamily\color{tsforestgreen}\thmname{#1}
 \nobreakspace\thmnumber{\@ifnotempty{#1}{}\@upn{#2}}
 \thmnote{\nobreakspace\the\thm@notefont\sffamily\bfseries\color{black} (#3)}}

\newtheoremstyle{styleexercise}
{0pt}{0pt}{\normalfont}{0pt}{\small\bf\sffamily\color{black}}{\;}{0.25em}
{\small\sffamily\color{black}\thmname{#1}
 \nobreakspace\thmnumber{\@ifnotempty{#1}{}\@upn{#2}}
 \thmnote{\nobreakspace\the\thm@notefont\sffamily\bfseries\color{black} (#3)}}

 \theoremstyle{styledefinition}
 \newtheorem{envdefinition}{Definition}[section]
 \theoremstyle{styletheorem}
 \newtheorem{envtheorem}{Theorem}[section]
 \newtheorem{envprop}{Proposition}[section]
 \newtheorem{envproprietes}{Propriety}[section]
 \newtheorem{envpropriete}{Proprieties}[section]
 \newtheorem{envproofpropriete}{Proof of propriety}[section]
 \newtheorem{envproofproprietes}{Proof of proprieties}[section]
 \newtheorem{envlemma}{Lemma}[section]
 \newtheorem{envprooflemma}{Proof of lemma}[section]
 \newtheorem{envproofprop}{Proof of proposition}[section]
 \theoremstyle{styleexercise}
 \newtheorem{envexercise}{Exercise}[section]
 \newtheorem{envproblem}{Problem}[section]
 \newtheorem{envprogramming}{Algorithm}[section]
 \newtheorem{envexample}{Example}[section]

 % Environments for theorems
 %--------------------------------------------------------------------------
 \newenvironment{definition}
                {\begin{boxdefinition}\begin{envdefinition}}
                {\end{envdefinition}\end{boxdefinition}}

 \newenvironment{theorem}
                {\begin{boxtheorem}\begin{envtheorem}}
                {\end{envtheorem}\end{boxtheorem}}

 \newenvironment{proposition}
               {\begin{boxtheorem}\begin{envprop}}
               {\end{envprop}\end{boxtheorem}}

 \newenvironment{corollary}
                {\begin{boxtheorem}\begin{envtheorem}}
                {\end{envtheorem}\end{boxtheorem}}

 \newenvironment{lemma}
                {\begin{boxtheorem}\begin{envlemma}}
                {\end{envlemma}\end{boxtheorem}}

 \newenvironment{propriete}
               {\begin{boxtheorem}\begin{envpropriete}}
               {\end{envpropriete}\end{boxtheorem}}

 \newenvironment{proprietes}
                {\begin{boxtheorem}\begin{envpropriete}}
                {\end{envpropriete}\end{boxtheorem}}

 \newenvironment{prooflemma}
               {\begin{boxprogramming}\begin{envprooflemma}}
               {\end{envprooflemma}\end{boxprogramming}}

 \newenvironment{proofprop}
               {\begin{boxprogramming}\begin{envproofprop}}
               {\end{envproofprop}\end{boxprogramming}}

 \newenvironment{proofpropriete}
               {\begin{boxprogramming}\begin{envproofpropriete}}
               {\end{envproofpropriete}\end{boxprogramming}}

 \newenvironment{proofproprietes}
               {\begin{boxprogramming}\begin{envproofproprietes}}
               {\end{envproofproprietes}\end{boxprogramming}}

 \newenvironment{exercise}
                {\begin{boxexercise}\begin{envexercise}}
                {\end{envexercise}\end{boxexercise}}

 \newenvironment{problem}
                {\begin{boxproblem}\begin{envproblem}}
                {\end{envproblem}\end{boxproblem}}

 \newenvironment{programming}
                {\begin{boxprogramming}\begin{envprogramming}}
                {\end{envprogramming}\end{boxprogramming}}

 \newenvironment{example}
                {\begin{boxexample}\begin{envexample}}
                {\end{envexample}\end{boxexample}}

% Other environments
%--------------------------------------------------------------------------
\newenvironment{summary}
{\hspace{10pt}\par\small\it
 \begin{list}{}{\leftmargin=35pt\rightmargin=25pt}
 \item\ignorespaces\advance\baselineskip -1pt}
{\end{list}\vspace{-0.5cm}}

\newenvironment{remark}
{\vspace{0.5cm}\noindent\it
 \marginpar{\vspace{-3mm}\includegraphics[width=1.0cm]{idea}}}
{\vspace{0.5cm}}

\newenvironment{walpha}
{\marginnote{\includegraphics[width=1.0cm]{wolframalpha}}}


% Bibliography - commented out by mav it does not work on overleaf
%--------------------------------------------------------------------------
\usepackage[backend=biber,defernumbers=true]{biblatex}
\addbibresource{bib_project.bib}

% Hyperref
%--------------------------------------------------------------------------

\RequirePackage[]{hyperref}
\hypersetup{
 pdfauthor       = { },
 colorlinks      = true,
 linkcolor       = tsforestgreen,
 linkbordercolor = tsforestgreen,
 urlcolor 	    = plum,
 citecolor		= plum,
 urlbordercolor  = plum,
 pdfstartview    = {XYZ = null null 1.00},
}

% Index
%--------------------------------------------------------------------------
\RequirePackage{makeidx}
\makeindex

% Packages for mathematics and algorithms
%--------------------------------------------------------------------------
%\RequirePackage{algorithm}
\RequirePackage{algpseudocode}
\RequirePackage{amsmath}
\RequirePackage{xfrac}
\RequirePackage{mathtools}
\RequirePackage{wasysym}
\RequirePackage{siunitx}
\newcommand\bmmax{2}
\RequirePackage{bm}
\RequirePackage{ bbold }

% Drawing
%--------------------------------------------------------------------------
\RequirePackage{pgf,tikz,pgfplots}
\pgfplotsset{compat=1.15}
\RequirePackage{mathrsfs}
\usetikzlibrary{arrows}

% Other useful packages
%--------------------------------------------------------------------------
\RequirePackage{blindtext}
\RequirePackage[most]{tcolorbox}
\RequirePackage{booktabs} %for beautiful tables
\RequirePackage{subcaption} %for table side by side with different caption
\RequirePackage{url}
\RequirePackage{marginnote}
\RequirePackage{multicol} %for multi column writting
\RequirePackage{float} %forcing figure to be there
\RequirePackage{subcaption} %for multiple caption when having more than one figure
\RequirePackage{enumitem} %change puce list
\setlist[description]{font=\textendash\enskip\scshape\bfseries}
\RequirePackage{pifont} %add ding for item

\usepackage{titling}

\usepackage{sectsty}
\sectionfont{\color{midnightblue}}
\subsectionfont{\color{royalblue}}
\subsubsectionfont{\color{periwinkle}}
\paragraphfont{\color{midnightblue}}
